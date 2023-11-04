# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.fixmatch import FixMatch
from .utils import CpMatchThresholdingHook
from semilearn.core.utils import (
    Bn_Controller,
    get_cosine_schedule_with_warmup,
    get_data_loader,
    get_dataset,
    get_optimizer,
)

@ALGORITHMS.register('cpmatch')
class CpMatch(AlgorithmBase):

    """
        CpMatch algorithm (https://arxiv.org/abs/2001.07685).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
    
    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
    
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(CpMatchThresholdingHook(alpha=0.1,delta=0.1), "ThresholdingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())


            # compute mask
            # mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
            mask = self.call_hook("masking", "ThresholdingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict
        
    def finetune(self):
        self.print_fn("Create finetuning optimizer and scheduler")
        # parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        # freeze all layers but the last fc
        if "vit" in self.args.net:
            linear_keyword = 'head'
        for name, param in self.model.named_parameters():
            if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                param.requires_grad = False
        self.optimizer = get_optimizer(
            self.model,
            'SGD',
            0.001,
            self.args.momentum,
            0.1,
            self.args.layer_decay,
        )
        epochs = 30
        warmup_epochs = 4
        per_epoch_steps = self.num_train_iter // self.epochs
        total_iters = self.num_train_iter + per_epoch_steps * (epochs + warmup_epochs)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 10*per_epoch_steps, num_warmup_steps=warmup_epochs*per_epoch_steps
        )
            
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")
        # print(self.dataset_dict.keys())
        for epoch in range(self.start_epoch, self.epochs+epochs):
            # self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= total_iters:
                break

            self.call_hook("before_train_epoch")

            for data_lb in self.loader_dict["all_lb"]:
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= total_iters:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.finetune_step(
                    **self.process_batch(**data_lb)
                )
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def finetune_step(self, x_lb, y_lb):

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            outs_x_lb = self.model(x_lb) 
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feat']
            feat_dict = {'x_lb':feats_x_lb}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            total_loss = sup_loss 

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item())
    
        return out_dict, log_dict
    
    def set_cali_optimizer(self):
        """
        set optimizer for algorithm
        """
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(
            self.model,
            self.args.optim,
            self.args.lr,
            self.args.momentum,
            self.args.weight_decay,
            self.args.layer_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.num_train_iter, num_warmup_steps=self.args.num_warmup_iter
        )
        return optimizer, scheduler

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]
