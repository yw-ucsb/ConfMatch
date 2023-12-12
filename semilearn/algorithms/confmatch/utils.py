# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
import numpy as np
import loralib as lora
from semilearn.algorithms.hooks import MaskingHook, PseudoLabelingHook
from scipy.optimize import brentq
from scipy.stats import binom
from sklearn.metrics import confusion_matrix
from semilearn.algorithms.softmatch.utils import SoftMatchWeightingHook

from semilearn.core.utils import ALGORITHMS, get_data_loader, send_model_cuda


class ConfMatchWeightingHook(SoftMatchWeightingHook):
    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if not self.prob_max_mu_t.is_cuda:
            self.prob_max_mu_t = self.prob_max_mu_t.to(logits_x_ulb.device)
        if not self.prob_max_var_t.is_cuda:
            self.prob_max_var_t = self.prob_max_var_t.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.update(algorithm, probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        # compute weight
        if not self.per_class:
            mu = algorithm.p_cutoff
            var = self.prob_max_var_t
        else:
            mu = self.prob_max_mu_t[max_idx]
            var = self.prob_max_var_t[max_idx]
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / (self.n_sigma ** 2))))
        return mask


class ConfMatchThresholdingHook(MaskingHook):
    """
    Dynamic Threshold in ConfMatch with conformal risk control;
    """
    def __init__(self, alpha, delta, n_lam_step=101, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Calibration dataset and loader;
        self.alpha = alpha
        self.delta = delta
        self.lam = np.linspace(0, 1, n_lam_step)
        self.cal_error_rate = 0.


    def update_dataset(self, pred_cali, algorithm):
        """
        Based on the lambda corresponding to the calculated UCB, release the calibration data back to the labeled
        dataset since the current model may not be good at predicting these data points right now;
        This is done by modifying the index in two Subsets: dset_lb and dset_cali and update the loader respectively;
        There are multiple ways to implement:
        1. One time release: once \lambda, or averaged \ldambda exceed a given threshold, release;
        2. Continuous release:

        Also, the released calibration might be kept in the calibration dataset -> this might not be a good choice;
        """
        pass

    def predict(self, algorithm):
        """
        Generating softmax score on the calibration dataset;
        Returns:
            o: output of the encoder;
            smx: the softmax score
        """
        loader_cali = algorithm.loader_dict["cali"]
        gpu = algorithm.gpu
        y_true = []
        o = []
        smx = []
        max_smx = []
        y_pred = []
        for data in loader_cali:
            x = data["x_lb"]
            y = data["y_lb"]

            if isinstance(x, dict):
                x = {k: v.cuda(gpu) for k, v in x.items()}
            else:
                x = x.cuda(gpu)
            y = y.cuda(gpu)

            o_batch = algorithm.model(x)["logits"]
            smx_batch = torch.nn.Softmax(dim=1)(o_batch)

            y_pred_batch = torch.argmax(smx_batch, dim=1)
            p_pred_batch, _ = smx_batch.max(axis=1)

            y_true.append(y)
            o.append(o_batch)
            smx.append(smx_batch)
            y_pred.append(y_pred_batch)
            max_smx.append(p_pred_batch)

        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()
        max_smx = torch.cat(max_smx).cpu().numpy()

        return {
            'o': o,
            'smx': smx,
            'max_smx': max_smx,
            'y_pred': y_pred,
            'y_true': y_true
        }

    def selective_control(self, pred_cali, delta):
        """
        The upper bound is constructed as: P(r^{+} > \hat_{r}) \le \delta;
        This tail probability can be bounded by concentration inequality, since R can be viewed as sum of i.i.d r;
        For Bernoulli random variables, the bound can be calculated exactly by calculating the CDF of Binomial distribution;
        """
        max_smx = pred_cali['max_smx']
        y_pred = pred_cali['y_pred']
        y_true = pred_cali['y_true']

        full_risk_cali = (y_pred != y_true).mean()
        self.alpha = full_risk_cali

        # Functions used to derive the UCB;
        def selective_risk(lam):
            idx_qualified = max_smx >= lam
            n_lam = idx_qualified.sum()
            n_failure = (y_pred[idx_qualified] != y_true[idx_qualified]).sum()
            return n_failure, n_lam

        def calculate_n_lam(lam):
            return (max_smx >= lam).sum() # TODO: n_lambda \le 0 can lead to bugs;

        def invert_for_ub(r, lam):
            n_failure, n_lam = selective_risk(lam)
            return binom.cdf(n_failure, n_lam, r) - delta

        def calculate_selective_risk_ucb(lam):
            return brentq(invert_for_ub, 0, 0.9999, args=(lam, )) # TODO: starting point should actually be full_risk_cali;

        # n_lam controls that at least 10 (can be others) calibration data points should have softmax score
        # greater than the lambda to be evaluated;
        lambdas = np.array([lam for lam in self.lam if calculate_n_lam(lam) >= 10])
        n_lambdas = np.array([calculate_n_lam(lam) for lam in self.lam if calculate_n_lam(lam) >= 10])

        assert len(lambdas) != 0, 'Not enough number of qualified calibration data, current{}, required 10.'.format(len(lambdas))

        print('all lams:', lambdas)
        print('all n_lams:', n_lambdas)

        try:
            for lhat in np.flip(lambdas):
                n_failure, n_lam = selective_risk(lhat)
                print('Current lam:', lhat, n_failure, n_lam)

                selective_risk_ucb = calculate_selective_risk_ucb(lhat - 1./lambdas.shape[0]) # TODO: finite sample correction can lead to bugs;
                if selective_risk_ucb > self.alpha:
                    print(f'Current model risk:{100 * full_risk_cali:.2f}%, threshold:{lhat:.2f}')
                    break
            p_cutoff = lhat.item()
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f'Failed control. Cal Error:{100 * full_risk_cali:.2f}%, threshold:0.95')
            p_cutoff = 0.95

        return p_cutoff, full_risk_cali

        # Currently deprecated!
        # # print(len(lambdas))
        # R = np.array([selective_risk(lam) for lam in lambdas])
        # # print(risks)
        # risk_min = risks.min() # TODO: this min risk can actually be not achievable;
        # gamma = self.gamma
        # self.cp_alpha = self.gamma * risk_model + (1. - self.gamma) * risk_min

    @torch.no_grad()
    def update(self, algorithm):
        # Calculate the predictions of the current model on the calibration dataset;
        pred_cali = self.predict(algorithm)
        # Update the cutoff threshold and the current model's risk;
        algorithm.p_cutoff, algorithm.cal_error_rate = self.selective_control(pred_cali, algorithm.delta)
        # Update the labeled dataset, calibration dataset and their corresponding loader;
        # Note: need to randomly reinitialize;

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if softmax_x_ulb:
            probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
        else:
            # logits are already probs
            probs_x_ulb = logits_x_ulb.detach()
        max_probs, _ = torch.max(probs_x_ulb, dim=-1)
        mask = max_probs.ge(algorithm.p_cutoff).to(max_probs.dtype)
        return mask

    @torch.no_grad()
    def before_train_step(self, algorithm, *args, **kwargs):
        # Update every 50 iterations;
        if (self.every_n_iters(algorithm, algorithm.num_log_iter) or algorithm.it == 0) and algorithm.it < algorithm.num_train_iter:
            # print(algorithm.num_log_iter, algorithm.it)
            self.update(algorithm)


    # def selective_control(self, algorithm):
    #     lambdas = np.linspace(0, 1, 101)
    #     model = algorithm.model
    #     cp_loader = algorithm.loader_dict["cali"]
    #     cal_labels = []
    #     cal_yhats = []
    #     cal_phats = []
    #     gpu = algorithm.gpu
    #     for data in cp_loader:
    #         x = data["x_lb"]
    #         y = data["y_lb"]
    #
    #         if isinstance(x, dict):
    #             x = {k: v.cuda(gpu) for k, v in x.items()}
    #         else:
    #             x = x.cuda(gpu)
    #         y = y.cuda(gpu)
    #
    #         o = model(x)["logits"]
    #
    #         smx = torch.nn.Softmax(dim=1)(o)
    #         # smx = algorithm.call_hook("dist_align", "DistAlignHook", probs_x_ulb=smx)
    #
    #         y_pred = torch.argmax(smx, dim=1)
    #         y_prob, _ = smx.max(axis=1)
    #         cal_labels.append(y)
    #         cal_yhats.append(y_pred)
    #         cal_phats.append(y_prob)
    #
    #     # Concatenate the list!
    #     cal_labels = torch.cat(cal_labels).cpu().numpy()
    #     cal_yhats = torch.cat(cal_yhats).cpu().numpy()
    #     cal_phats = torch.cat(cal_phats).cpu().numpy()
    #
    #     cal_error_rate = (cal_labels != cal_yhats).mean()
    #
    #     # confusion matrix normalized by ground truth
    #     cf_mat = confusion_matrix(cal_labels, cal_yhats, normalize="true")
    #     cf_mat = torch.tensor(cf_mat).cuda(gpu)
    #
    #     # confusion matrix normalized by prediction
    #     cf_mat_pred = confusion_matrix(cal_labels, cal_yhats, normalize="pred")
    #     cf_mat_pred = torch.tensor(cf_mat_pred).cuda(gpu)
    #
    #     # Define selective risk
    #     def selective_risk(lam):
    #         return (cal_yhats[cal_phats >= lam] != cal_labels[cal_phats >= lam]).sum()/(cal_phats >= lam).sum()
    #
    #     def nlambda(lam):
    #         return (cal_phats > lam).sum() #TODO: n_lambda \le 0 can lead to bugs;
    #
    #     # The upper bound is constructed as: P(r^{+} > \hat_{r}) \le \delta;
    #     # This tail probability can be bounded by concentration inequality, since R can be viewed as sum of i.i.d r;
    #     # For Bernoulli random variables, the bound can be calculated exactly by calculating the CDF of Binomial distribution;
    #     # Now, given a particular \hat{R}, multiple r^{+} can be found with
    #     def invert_for_ub(r, lam):
    #         return binom.cdf(selective_risk(lam)*nlambda(lam), nlambda(lam), r)-self.cp_delta
    #
    #     # Construct upper bound
    #     def selective_risk_ub(lam):
    #         return brentq(invert_for_ub, 0, 0.9999, args=(lam,))
    #
    #     # Compute the smallest risk;
    #     # n_lambda controls that at least 10 (can be others) calibration data points should have softmax score
    #     # greater than the lambda to be evaluated;
    #     lambdas = np.array([lam for lam in lambdas if nlambda(lam) >= 10]) # Make sure there's some data in the top bin.
    #     # print(len(lambdas))
    #     risks = np.array([selective_risk(lam) for lam in lambdas])
    #     # print(risks)
    #     risk_min = risks.min()
    #     gamma = self.gamma
    #     self.cp_alpha = gamma * cal_error_rate + (1-gamma) * risk_min
    #     # print(f'Cal Error:{100*cal_error_rate:.2f}%, min risk:{100*risk_min:.2f}%, alpha:{100*self.cp_alpha:.2f}')
    #     # Scan to choose lambda hat;
    #     try:
    #         for lhat in np.flip(lambdas):
    #             # print('lhat: ',lhat, lhat-1/lambdas.shape[0])
    #             risk = selective_risk_ub(lhat-1/lambdas.shape[0]) # TODO: finite sample correction can lead to bugs;
    #             # print('risk: ',risk, lhat, lambdas.shape[0])
    #             if risk > self.cp_alpha:
    #                 print(f'Cal Error:{100*cal_error_rate:.2f}%, min risk:{100*risk_min:.2f}%, alpha:{100*self.cp_alpha:.2f}, threshold:{lhat:.2f}')
    #                 break
    #         return lhat.item(), cal_error_rate, cf_mat, cf_mat_pred
    #     except:
    #         print(f'Failed control. Cal Error:{100*cal_error_rate:.2f}%, min risk:{100*risk_min:.2f}%, alpha:{100*self.cp_alpha:.2f}, threshold:0.95')
    #         return 0.95, cal_error_rate, cf_mat, cf_mat_pred
    #
    # @torch.no_grad()
    # def update(self, algorithm):
    #     algorithm.p_cutoff, algorithm.cal_error_rate, algorithm.cf_mat, algorithm.cf_mat_pred = self.selective_control(algorithm)
    #
    # @torch.no_grad()
    # def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
    #     if softmax_x_ulb:
    #         # probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
    #         probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
    #     else:
    #         # logits is already probs
    #         probs_x_ulb = logits_x_ulb.detach()
    #     max_probs, _ = torch.max(probs_x_ulb, dim=-1)
    #     mask = max_probs.ge(algorithm.p_cutoff).to(max_probs.dtype)
    #     return mask


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


def create_lora_ft_vit(args, model, r=8):
    # Find attention layers in vit and replace the q, v with lora linear layer;
    # Note USB implements qkv with a single linear layer;
    # In USB, attention layers are marked as sel.blocks[i].attn.qkv;
    r = 2
    n_feat = model.num_features
    n_classes = model.num_classes

    # Setup trainable parameters;
    state_dict = model.state_dict()
    for block in model.blocks:
        block.attn.qkv = lora.MergedLinear(n_feat, 3*n_feat, r=r, enable_lora=[True, False, True])

    # Find the final head and replace with lora linear layers;
    # model.head = lora.Linear(n_feat, n_classes, r=r)
    model.load_state_dict(state_dict, strict=False)
    # lora.mark_only_lora_as_trainable(model, bias='all')
    lora.mark_only_lora_as_trainable(model)
    for name, param in model.named_parameters():
        if name in ['%s.weight' % 'head', '%s.bias' % 'head']:
            param.requires_grad = True
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
