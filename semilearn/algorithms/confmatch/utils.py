# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
import numpy as np
import loralib as lora
from semilearn.algorithms.hooks import MaskingHook, PseudoLabelingHook
from scipy.optimize import brentq
from scipy.stats import binom
from sklearn.metrics import confusion_matrix

from semilearn.core.utils import ALGORITHMS, get_data_loader, send_model_cuda

class ConfMatchThresholdingHook(MaskingHook):
    """
    Dynamic Threshold in CpMatch
    """
    def __init__(self, alpha, delta, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cp_alpha = alpha
        self.cp_delta = delta
        self.gamma = gamma
        self.cal_error_rate = 0.

    def selective_control(self, algorithm):
        lambdas = np.linspace(0, 1, 101)
        model = algorithm.model
        cp_loader = algorithm.loader_dict["cali"]
        cal_labels = []
        cal_yhats = []
        cal_phats = []
        gpu = algorithm.gpu
        for data in cp_loader:
            x = data["x_lb"]
            y = data["y_lb"]

            if isinstance(x, dict):
                x = {k: v.cuda(gpu) for k, v in x.items()}
            else:
                x = x.cuda(gpu)
            y = y.cuda(gpu)

            o = model(x)["logits"]

            smx = torch.nn.Softmax(dim=1)(o)
            y_pred = torch.argmax(smx, dim=1)
            y_prob, _ = smx.max(axis=1)
            cal_labels.append(y)
            cal_yhats.append(y_pred)
            cal_phats.append(y_prob)

        # Concatenate the list!
        cal_labels = torch.cat(cal_labels).cpu().numpy()
        cal_yhats = torch.cat(cal_yhats).cpu().numpy()
        cal_phats = torch.cat(cal_phats).cpu().numpy()
        # print(len(cal_labels))
        # print(cal_labels[:6])
        # print(cal_yhats[:6])
        # Compute calibration error rate
        cal_error_rate = (cal_labels != cal_yhats).mean()

        # confusion matrix normalized by ground truth
        cf_mat = confusion_matrix(cal_labels, cal_yhats, normalize="true")
        cf_mat = torch.tensor(cf_mat).cuda(gpu)

        # confusion matrix normalized by prediction
        cf_mat_pred = confusion_matrix(cal_labels, cal_yhats, normalize="pred")
        cf_mat_pred = torch.tensor(cf_mat_pred).cuda(gpu)

        # Define selective risk
        def selective_risk(lam): return (cal_yhats[cal_phats >= lam] != cal_labels[cal_phats >= lam]).sum()/(cal_phats >= lam).sum()
        def nlambda(lam): return (cal_phats > lam).sum()
        def invert_for_ub(r, lam): return binom.cdf(selective_risk(lam)*nlambda(lam), nlambda(lam), r)-self.cp_delta
        # Construct upper boud
        def selective_risk_ub(lam): return brentq(invert_for_ub, 0, 0.9999, args=(lam,))

        # Compute the smallest risk
        lambdas = np.array([lam for lam in lambdas if nlambda(lam) >= 10]) # Make sure there's some data in the top bin.
        # print(len(lambdas))
        risks = np.array([selective_risk(lam) for lam in lambdas])
        # print(risks)
        risk_min = risks.min()
        gamma = self.gamma
        self.cp_alpha = gamma * cal_error_rate + (1-gamma) * risk_min
        # print(f'Cal Error:{100*cal_error_rate:.2f}%, min risk:{100*risk_min:.2f}%, alpha:{100*self.cp_alpha:.2f}')
        # Scan to choose lamabda hat
        try:
            for lhat in np.flip(lambdas):
                # print('lhat: ',lhat, lhat-1/lambdas.shape[0])
                risk = selective_risk_ub(lhat-1/lambdas.shape[0])
                # print('risk: ',risk, lhat, lambdas.shape[0])
                if risk > self.cp_alpha: 
                    print(f'Cal Error:{100*cal_error_rate:.2f}%, min risk:{100*risk_min:.2f}%, alpha:{100*self.cp_alpha:.2f}, threshold:{lhat:.2f}')
                    break
            return lhat.item(), cal_error_rate, cf_mat, cf_mat_pred
        except:
            print(f'Failed control. Cal Error:{100*cal_error_rate:.2f}%, min risk:{100*risk_min:.2f}%, alpha:{100*self.cp_alpha:.2f}, threshold:0.95')
            return 0.95, cal_error_rate, cf_mat, cf_mat_pred
    
    @torch.no_grad()
    def update(self, algorithm):
        algorithm.p_cutoff, algorithm.cal_error_rate, algorithm.cf_mat, algorithm.cf_mat_pred = self.selective_control(algorithm)

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if softmax_x_ulb:
            # probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
            probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()
        max_probs, _ = torch.max(probs_x_ulb, dim=-1)
        mask = max_probs.ge(algorithm.p_cutoff).to(max_probs.dtype)
        return mask


    # @torch.no_grad()
    # def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):

    #     # update the threshould: algorithm.p_cutoff
    #     self.update(algorithm)

    #     if softmax_x_ulb:
    #         # probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
    #         probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
    #     else:
    #         # logits is already probs
    #         probs_x_ulb = logits_x_ulb.detach()
    #     max_probs, _ = torch.max(probs_x_ulb, dim=-1)
    #     mask = max_probs.ge(algorithm.p_cutoff).to(max_probs.dtype)
    #     return mask

    @torch.no_grad()
    def before_train_step(self, algorithm, *args, **kwargs):

        # update the threshould: algorithm.p_cutoff, every 50 iterations
        if (self.every_n_iters(algorithm, algorithm.num_log_iter) or algorithm.it == 0) and algorithm.it <algorithm.num_train_iter:
            print(algorithm.num_log_iter, algorithm.it)
            self.update(algorithm)


def create_lora_ft_vit(args, model, r=8):
    # Find attention layers in vit and replace the q, v with lora linear layer;
    # Note USB implements qkv with a single linear layer;
    # In USB, attention layers are marked as sel.blocks[i].attn.qkv;
    n_feat = model.num_features
    n_classes = model.num_classes
    for block in model.blocks:
        block.attn.qkv = lora.MergedLinear(n_feat, 3*n_feat, r=r, enable_lora=[True, False, True])

    # Find the final head and replace with lora linear layers;
    # model.head = lora.Linear(n_feat, n_classes, r=r)
    # Setup trainable parameters;
    lora.mark_only_lora_as_trainable(model)
    # Send model to devices;
    model = send_model_cuda(args, model)

    return model


def create_vanilla_ft_vit(model):
    for name, param in model.named_parameters():
        if name not in ['%s.weight' % 'head', '%s.bias' % 'head']:
            param.requires_grad = False


class ConfMatchSoftPseudoLabelingHook(PseudoLabelingHook):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def gen_ulb_targets(self, 
                        algorithm, 
                        logits, 
                        use_hard_label=True, 
                        T=1.0,
                        softmax=True, # whether to compute softmax for logits, input must be logits
                        label_smoothing=0.0):
        
        """
        generate pseudo-labels from logits/probs

        Args:
            algorithm: base algorithm
            logits: logits (or probs, need to set softmax to False)
            use_hard_label: flag of using hard labels instead of soft labels
            T: temperature parameters
            softmax: flag of using softmax on logits
            label_smoothing: label_smoothing parameter
        """

        logits = logits.detach()
        # if use_hard_label:
        #     # return hard label directly
        #     pseudo_label = torch.argmax(logits, dim=-1)
        #     if label_smoothing:
        #         pseudo_label = smooth_targets(logits, pseudo_label, label_smoothing)
        #     return pseudo_label
        
        # return soft label
        if softmax:
            # pseudo_label = torch.softmax(logits / T, dim=-1)
            pseudo_label = algorithm.compute_prob(logits / T)
        else:
            # inputs logits converted to probabilities already
            pseudo_label = logits
        
        return pseudo_label
