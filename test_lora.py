
import argparse
import logging
import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
from semilearn.algorithms import get_algorithm, name2alg
from semilearn.core.utils import (
    TBLog,
    count_parameters,
    get_logger,
    get_net_builder,
    get_port,
    over_write_args_from_file,
    send_model_cuda,
)
from semilearn.imb_algorithms import get_imb_algorithm, name2imbalg


def get_config():
    from semilearn.algorithms.utils import str2bool

    parser = argparse.ArgumentParser(description="Semi-Supervised Learning (USB)")

    """
    Saving & loading of the model.
    """
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("-sn", "--save_name", type=str, default="fixmatch")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_path", type=str)
    parser.add_argument("-o", "--overwrite", action="store_true", default=True)
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        help="Use tensorboard to plot and save curves",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use wandb to plot and save curves"
    )
    parser.add_argument(
        "--use_aim", action="store_true", help="Use aim to plot and save curves"
    )

    """
    Training Configuration of FixMatch
    """
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument(
        "--num_train_iter",
        type=int,
        default=20,
        help="total number of training iterations",
    )
    parser.add_argument(
        "--num_warmup_iter", type=int, default=0, help="cosine linear warmup iterations"
    )
    parser.add_argument(
        "--num_eval_iter", type=int, default=10, help="evaluation frequency"
    )
    parser.add_argument("--num_log_iter", type=int, default=5, help="logging frequency")
    parser.add_argument("-nl", "--num_labels", type=int, default=400)
    parser.add_argument("-bsz", "--batch_size", type=int, default=8)
    parser.add_argument(
        "--uratio",
        type=int,
        default=1,
        help="the ratio of unlabeled data to labeled data in each mini-batch",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="batch size of evaluation data loader (it does not affect the accuracy)",
    )
    parser.add_argument(
        "--ema_m", type=float, default=0.999, help="ema momentum for eval_model"
    )
    parser.add_argument("--ulb_loss_ratio", type=float, default=1.0)

    """
    Optimizer configurations
    """
    parser.add_argument("--optim", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=1.0,
        help="layer-wise learning rate decay, default to 1.0 which means no layer "
        "decay",
    )

    """
    Backbone Net Configurations
    """
    parser.add_argument("--net", type=str, default="wrn_28_2")
    parser.add_argument("--net_from_name", type=str2bool, default=False)
    parser.add_argument("--use_pretrain", default=False, type=str2bool)
    parser.add_argument("--pretrain_path", default="", type=str)

    """
    Algorithms Configurations
    """

    ## core algorithm setting
    parser.add_argument(
        "-alg", "--algorithm", type=str, default="fixmatch", help="ssl algorithm"
    )
    parser.add_argument(
        "--use_cat", type=str2bool, default=True, help="use cat operation in algorithms"
    )
    parser.add_argument(
        "--amp",
        type=str2bool,
        default=False,
        help="use mixed precision training or not",
    )
    parser.add_argument("--clip_grad", type=float, default=0)

    ## imbalance algorithm setting
    parser.add_argument(
        "-imb_alg",
        "--imb_algorithm",
        type=str,
        default=None,
        help="imbalance ssl algorithm",
    )

    """
    Data Configurations
    """

    ## standard setting configurations
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("-ds", "--dataset", type=str, default="cifar10")
    parser.add_argument("-nc", "--num_classes", type=int, default=10)
    parser.add_argument("--train_sampler", type=str, default="RandomSampler")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--include_lb_to_ulb",
        type=str2bool,
        default="True",
        help="flag of including labeled data into unlabeled data, default to True",
    )

    ## imbalanced setting arguments
    parser.add_argument(
        "--lb_imb_ratio",
        type=int,
        default=1,
        help="imbalance ratio of labeled data, default to 1",
    )
    parser.add_argument(
        "--ulb_imb_ratio",
        type=int,
        default=1,
        help="imbalance ratio of unlabeled data, default to 1",
    )
    parser.add_argument(
        "--ulb_num_labels",
        type=int,
        default=None,
        help="number of labels for unlabeled data, used for determining the maximum "
        "number of labels in imbalanced setting",
    )

    ## cv dataset arguments
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--crop_ratio", type=float, default=0.875)

    ## nlp dataset arguments
    parser.add_argument("--max_length", type=int, default=512)

    ## speech dataset algorithms
    parser.add_argument("--max_length_seconds", type=float, default=4.0)
    parser.add_argument("--sample_rate", type=int, default=16000)

    """
    multi-GPUs & Distributed Training
    """

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)  # noqa: E501
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="**node rank** for distributed training"
    )
    parser.add_argument(
        "-du",
        "--dist-url",
        default="tcp://127.0.0.1:11111",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--seed", default=1, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--multiprocessing_distributed",
        type=str2bool,
        default=False,
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )

    """
    Finetuning arguments;
    """
    parser.add_argument("--ft_optim", type=str, default="SGD")
    parser.add_argument("--ft_lr", type=float, default=3e-2)
    parser.add_argument("--ft_momentum", type=float, default=0.9)
    parser.add_argument("--ft_weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--ft_layer_decay",
        type=float,
        default=1.0,
        help="layer-wise learning rate decay, default to 1.0 which means no layer "
             "decay",
    )
    parser.add_argument("--ft_epoch", type=int, default=1)
    parser.add_argument(
        "--_num_ft_iter",
        type=int,
        default=20,
        help="total number of training iterations",
    )
    parser.add_argument(
        "--ft_num_warmup_iter", type=int, default=0, help="cosine linear warmup iterations"
    )
    parser.add_argument("-ft_bsz", "--ft_batch_size", type=int, default=64)

    # config file
    parser.add_argument("--c", type=str, default='./test_lora_config.yaml')

    # add algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    for argument in name2alg[args.algorithm].get_argument():
        parser.add_argument(
            argument.name,
            type=argument.type,
            default=argument.default,
            help=argument.help,
        )

    # add imbalanced algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    if args.imb_algorithm is not None:
        for argument in name2imbalg[args.imb_algorithm].get_argument():
            parser.add_argument(
                argument.name,
                type=argument.type,
                default=argument.default,
                help=argument.help,
            )
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    # print(args.gpu)
    # print("dataset:", args.dataset)
    # print(args.confmatch_cali_s)
    # print(args.lambda_conf, args.confmatch_gamma, args.conf_loss)
    return args


def create_model(args):
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None

    logger = get_logger(args.save_name, save_path, logger_level)
    _net_builder = get_net_builder(args.net, args.net_from_name)

    model = get_algorithm(args, _net_builder, tb_log, logger)

    # SET Devices for (Distributed) DataParallel
    model.model = send_model_cuda(args, model.model)
    model.ema_model = send_model_cuda(args, model.ema_model, clip_batch=False)

    return model



if __name__ == '__main__':
    # Create args for the model, note to specify resume and load path;
    args = get_config()


    # Create model and load vanilla model;
    model = create_model(args=args)
    saved_model = torch.load('/home/y_yin/SSL-Benchmark-USB/saved_models/usb_cv/cpmatch/cifar100_400_0_True_0.1_0.1_1.00_cali5_confTrue_0.001/latest_model.pth')
    model.model.load_state_dict(saved_model['model'])
    model.ema_model.load_state_dict(saved_model['ema_model'])
    model.it = saved_model['it']
    model.start_epoch = saved_model['epoch']
    model.epoch = saved_model['epoch']
    # Test the evaluation function;
    # Not in this code, ema is not initialized with hook before the run, so we set an additional if else in evaluate();
    # This should be removed after testing lora;
    # eval_dict_vanilla = model.evaluate()
    # Test the lora finetuning function;
    logging.info('Finetuning model with Lora...')
    model.finetune()
    # Test the final performance after finetuning;
    logging.info('Test Lora finetuned model...')
    eval_dict_vanilla = model.evaluate()

