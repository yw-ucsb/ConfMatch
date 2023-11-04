# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
import numpy as np
from semilearn.algorithms.hooks import MaskingHook
from scipy.optimize import brentq
from scipy.stats import binom

class CpMatchThresholdingHook(MaskingHook):
    """
    Dynamic Threshold in CpMatch
    """
    def __init__(self, alpha, delta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cp_alpha = alpha
        self.cp_delta = delta

    def selective_control(self, algorithm):
        lambdas = np.linspace(0,1,101)
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

        # Define selective risk
        def selective_risk(lam): return (cal_yhats[cal_phats >= lam] != cal_labels[cal_phats >= lam]).sum()/(cal_phats >= lam).sum()
        def nlambda(lam): return (cal_phats > lam).sum()
        def invert_for_ub(r,lam): return binom.cdf(selective_risk(lam)*nlambda(lam),nlambda(lam),r)-self.cp_delta
        # Construct upper boud
        def selective_risk_ub(lam): return brentq(invert_for_ub,0,0.9999,args=(lam,))

        # Compute the smallest risk
        lambdas = np.array([lam for lam in lambdas if nlambda(lam) >= 10]) # Make sure there's some data in the top bin.
        # print(len(lambdas))
        risks = np.array([selective_risk(lam) for lam in lambdas])
        # print(risks)
        risk_min = risks.min()
        gamma = 0.5
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
                    return lhat.item()
        except:
            print(f'Failed control. Cal Error:{100*cal_error_rate:.2f}%, min risk:{100*risk_min:.2f}%, alpha:{100*self.cp_alpha:.2f}, threshold:0.95')
            return 0.95
    
    @torch.no_grad()
    def update(self, algorithm):
        algorithm.p_cutoff = self.selective_control(algorithm)

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
        if self.every_n_iters(algorithm, algorithm.num_log_iter) or algorithm.it == 0:
            print(algorithm.num_log_iter, algorithm.it)
            self.update(algorithm)

