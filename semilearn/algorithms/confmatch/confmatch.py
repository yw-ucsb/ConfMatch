# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import loralib as lora
from torch.nn import LayerNorm
import torch.nn.functional as F
import numpy as np
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS, get_data_loader, send_model_cuda, count_parameters
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.fixmatch import FixMatch
from .utils import ConfMatchThresholdingHook, create_lora_ft_vit, create_vanilla_ft_vit, ConfMatchSoftPseudoLabelingHook, ConfMatchWeightingHook
from torchsummary import summary
from semilearn.core.utils import (
    Bn_Controller,
    get_cosine_schedule_with_warmup,
    get_data_loader,
    get_dataset,
    get_optimizer,
)

from torch.utils.data import DataLoader, random_split, Subset
from semilearn.datasets import get_collactor, name2sampler, DistributedSampler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm


class ConfBase(AlgorithmBase):
    """
    New base from AlgorithmBase with support of finetuning;
    """
    def __init__(self, args, net_builder, tb_log, logger):
        super().__init__(args, net_builder, tb_log, logger)
        # Setup finetuning related arguments here; TODO: Make sure the start / end epoch / iterations are correctly configured;
        self.status = None
        self.ft_start_epoch = args.epoch
        self.ft_epoch = args.ft_epoch
        self.num_ft_iter = args.num_ft_iter
        self.ft_num_warmup_iter = args.ft_num_warmup_iter

        self.total_iter = self.num_train_iter + args.num_ft_iter

    def set_finetuning(self):
        """
        Set up finetuning model with LORA and corresponding optimizer, scheduler;
        Note: this will completely overwrite the original optimizer and scheduler used during the training phase,
        save before doing so!
        """
        self.print_fn("Create fine-tuning model, optimizer and scheduler...")
        # Create Lora model;
        # self.model = create_lora_ft_vit(self.args, self.model)
        # summary(self.model)
        # print(self.model)
        # assert False
        create_vanilla_ft_vit(self.model)
        print(f"Number of Trainable Params: {count_parameters(self.model)}")

        self.optimizer = get_optimizer(
            self.model,
            self.args.ft_optim,
            self.args.ft_lr,
            self.args.ft_momentum,
            self.args.ft_weight_decay,
            self.args.ft_layer_decay,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, self.args.num_ft_iter, num_warmup_steps=self.ft_num_warmup_iter
        )


    def finetune(self):
        """
        Finetuning logs will be appended to the end of the training;
        """
        self.set_finetuning()

        self.model.train()
        self.call_hook("before_run")
        # Disable EMA by resetting ema to None to avoid update;
        self.ema = None

        for epoch in tqdm(range(self.ft_start_epoch, self.ft_start_epoch + self.ft_epoch)):
            self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.total_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb in self.loader_dict["ft"]:
                # prevent the training iterations exceed total_iter;
                if self.it >= self.total_iter:
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
        # Fintuning is done with CE loss on all labeled data;
        # print(self.model.head)
        with self.amp_cm():
            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feat']
            feat_dict = {'x_lb': feats_x_lb}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            total_loss = sup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item())

        return out_dict, log_dict


@ALGORITHMS.register('confmatch')
class ConfMatch(ConfBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        # Initialized algorithm specified arguments;
        # This has to be before super method since dataloader creation requires n_repeat_loader_cali arg;
        self.T = args.T
        self.p_cutoff = args.p_cutoff
        self.use_hard_label = args.hard_label
        self.n_repeat_loader_cali = args.n_repeat_loader_cali
        self.alpha = args.confmatch_alpha
        self.delta = args.confmatch_delta
        self.gamma = args.confmatch_gamma
        self.cal_error_rate = 0.
        self.cf_mat = torch.zeros((2, 2))
        self.cf_mat_pred = torch.zeros((2, 2))
        self.conf_loss = args.conf_loss
        self.lambda_conf = args.lambda_conf

        self.top5_metric = MulticlassAccuracy(num_classes=args.num_classes, top_k=5)

        super().__init__(args, net_builder, tb_log, logger)

    def set_data_loader(self):
        """
        set loader_dict;
        """
        # Call Base class buildup;
        loader_dict = AlgorithmBase.set_data_loader(self)

        # CpMatch Loader;
        # Loader that repeats loading the full calibration set for k times for expansion;
        # Note that each loading will utilize different w augmentation; TODO: check RandCrop!

        dset_cali = self.dataset_dict["cali"]
        n_cali = len(dset_cali)

        num_samples = self.n_repeat_loader_cali * n_cali

        loader_cali = DataLoader(dset_cali, batch_size=n_cali, shuffle=False, num_workers=self.args.num_workers,
                                 sampler=DistributedSampler(dset_cali, num_replicas=1, rank=0, num_samples=num_samples),
                                 drop_last=False)

        loader_dict["cali"] = loader_cali

        loader_dict["ft"] = get_data_loader(
            self.args,
            self.dataset_dict["ft"],
            self.args.ft_batch_size,
            data_sampler=self.args.train_sampler,
            num_iters=self.args.num_ft_iter,
            num_epochs=self.args.ft_epoch,
            num_workers=self.args.num_workers,
            distributed=self.distributed,
        )

        return loader_dict

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(ConfMatchThresholdingHook(alpha=self.alpha, delta=self.delta, gamma=self.gamma), "ThresholdingHook")
        self.register_hook(ConfMatchSoftPseudoLabelingHook(), "SoftPseudoLabelingHook")
        # self.register_hook(ConfMatchWeightingHook(num_classes=self.num_classes, n_sigma=self.args.n_sigma, momentum=self.args.ema_p, per_class=self.args.per_class), "MaskingHook")
        # self.register_hook(
        #     DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='uniform' if self.args.dist_uniform else 'model'),
        #     "DistAlignHook")
        super().set_hooks()

    def cpmatch_contrastive_loss(self, x_lb, y_lb, T=0.2):
        # embedding similarity;
        n = x_lb.shape[0]
        I = torch.eye(n, device=x_lb.device)
        
        x_lb_normed = F.normalize(x_lb, p=2, dim=1)
        sim = torch.mm(x_lb_normed, x_lb_normed.t().detach())
        # sim_probs = sim / sim.sum(1, keepdim=True)
        idx_row, idx_col = torch.meshgrid(y_lb, y_lb)
        M = self.cf_mat[idx_row, idx_col]
        # contrastive loss
        flags = torch.eq(idx_col, idx_row).int() - I

        M = M - flags
        
        loss = (sim * M/2).sum()

        return loss

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, y_ulb):
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
            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            # uniform distribution alignment 
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)


            # compute mask based on algorithm defined thresholding;
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)
            use_soft_label = False
            if use_soft_label:
                unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                                self.cf_mat_pred[:,pseudo_label].T,
                                                'ce',
                                                mask=mask)

            else:
                unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                                pseudo_label,
                                                'ce',
                                                mask=mask)

            # Calculate the accuracy of the pseudo labels on selected unlabeled data;
            ulb_select_top1 = torch.eq(pseudo_label, y_ulb).float() * mask # TODO
            ulb_select_top1 = ulb_select_top1.sum() / mask.shape[0]

            # Confusion matrix regularization loss;
            if self.conf_loss:
                conf_loss = self.cpmatch_contrastive_loss(feats_x_lb, y_lb)
            else:
                conf_loss = torch.tensor(0., device=x_lb.device)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_conf * conf_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         conf_loss=conf_loss.item(),
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item(),
                                         threshold=self.p_cutoff,
                                         alpha=self.hooks_dict["ThresholdingHook"].cp_alpha,
                                         cali_acc=1-self.cal_error_rate,
                                         ulb_select_top1=ulb_select_top1.item(),
                                         )
        return out_dict, log_dict

    def evaluate(self, eval_dest="eval", out_key="logits", return_logits=False):
        """
        evaluation function
        """
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        # y_probs = []
        y_logits = []
        with torch.no_grad():
            for data in eval_loader:
                # Load data and send to device;
                x = data["x_lb"]
                y = data["y_lb"]
                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                # Calculate batch level metrics;
                logits = self.model(x)[out_key]
                loss = F.cross_entropy(logits, y, reduction="mean", ignore_index=-1)

                # Merge batch level metric into a list, or sum to be processed later after iterating full set;
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                total_loss += loss.item() * num_batch

        # Calculate dataset level metrics;
        # Convert list into np arrays;
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)

        # Calculate dataset metrics here;
        # Default metrics from usb;
        top1 = accuracy_score(y_true, y_pred)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        F1 = f1_score(y_true, y_pred, average="macro")
        cf_mat = confusion_matrix(y_true, y_pred, normalize="true")
        self.print_fn("confusion matrix:\n" + np.array_str(cf_mat))

        # Other metrics we want to monitor;
        top5 = self.top5_metric(torch.tensor(y_logits), torch.tensor(y_true)).item()

        if self.ema is not None:
            self.ema.restore()

        self.model.train()

        eval_dict = {
            eval_dest + "/loss": total_loss / total_num,
            eval_dest + "/top-1-acc": top1,
            eval_dest + "/balanced_acc": balanced_top1,
            eval_dest + "/precision": precision,
            eval_dest + "/recall": recall,
            eval_dest + "/F1": F1,
            eval_dest + "/top-5-acc": top5,
        }
        if return_logits:
            eval_dict[eval_dest + "/logits"] = y_logits
        return eval_dict
    
    @staticmethod
    def get_argument():
        return [
            # FixMatch required arguments;
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            # ConfMatch specified arguments;
            # Risk control related arguments;
            SSL_Argument("--num_cali", default=100, type=int, help="Number of calibration data points"),
            SSL_Argument("--n_repeat_loader_cali", default=5, type=int,
                         help="repeat x times of weak argumentation for calibration data"),
            SSL_Argument("--confmatch_alpha", default=.1, type=float, help="hyper-parameter: error rate"),
            SSL_Argument("--confmatch_delta", default=.1, type=float, help="hyper-parameter: failure rate"),
            SSL_Argument("--confmatch_gamma", default=.5, type=float,
                         help="hyper-parameter: weight of cali PL accuracy"),
            # Confusion matrix related arguments;
            SSL_Argument("--conf_loss", default=False, type=bool, help="Confusion matrix loss"),
            SSL_Argument("--lambda_conf", default=0.001, type=float, help="Weight of confusion matrix loss"),

            # Archived;
            SSL_Argument("--confmatch_cali_s", default=False, type=bool,
                         help="Strong argumented calibration data for training"),
            # Softmatch
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--dist_uniform', str2bool, True),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--n_sigma', int, 2),
            SSL_Argument('--per_class', str2bool, False),
        ]
