data_dir="/data/datasets/usb"
load_path="./saved_models/usb_cv/cpmatch_cifar100_400_0_False/latest_model.pth"
save_dir="./saved_models/usb_cv/cpmatch_cifar100_400_0_finetune/"
gpu=7
seeds=(0)
dataset="cifar100_400"
method="cpmatch"
include_lb_to_ulb="True"
use_wandb=""
# use_wandb="--use_wandb"
# method="fixmatch"
# dataset="cifar100_300"
for seed in "${seeds[@]}";do
cmd="
python train.py --c config/usb_cv/${method}/${method}_${dataset}_0.yaml 
--gpu=$gpu 
--seed=$seed
--save_name=${method}_${dataset}_${seed}_${include_lb_to_ulb}
--data_dir=$data_dir
--multiprocessing_distributed=False
$use_wandb
--include_lb_to_ulb=$include_lb_to_ulb
--save_dir=$save_dir
--load_path=$load_path
"
$cmd
echo $cmd
done