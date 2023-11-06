data_dir="/data/datasets/usb"
load_path="--load_path=./saved_models/usb_cv/cpmatch_cifar100_400_0_False/latest_model.pth"
save_dir="--save_dir=./saved_models/usb_cv/cpmatch_cifar100_400_0_finetune/"
save_dir="--save_dir=./saved_models/usb_cv/"
load_path="--load_path=./saved_models/usb_cv/"
gpu=6
seeds=(0)
dataset="cifar100_400"
dataset="tissuemnist_400"
method="cpmatch"
# method="fixmatch"
include_lb_to_ulb="False"
include_lb_to_ulb="True"
use_wandb=""
use_wandb="--use_wandb"
n_repeat_loader_cali=5
confmatch_alpha=0.1
confmatch_delta=0.1
confmatch_gamma=0.50
# cali_s="True"
cali_s="False"
# dataset="cifar100_300"
for seed in "${seeds[@]}";do
save_name="${method}_${dataset}_${seed}_${include_lb_to_ulb}_${confmatch_alpha}_${confmatch_delta}_${confmatch_gamma}_cali${n_repeat_loader_cali}_${cali_s}"
# save_name="${method}_${dataset}_${seed}_${include_lb_to_ulb}"
cmd="
python train.py --c config/usb_cv/${method}/${method}_${dataset}_0.yaml 
--gpu=$gpu 
--seed=$seed
--save_name=$save_name
--data_dir=$data_dir
--multiprocessing_distributed=False
--n_repeat_loader_cali=$n_repeat_loader_cali
--confmatch_alpha=$confmatch_alpha
--confmatch_delta=$confmatch_delta
--confmatch_gamma=$confmatch_gamma
$use_wandb
--include_lb_to_ulb=$include_lb_to_ulb
$save_dir
$load_path
"
$cmd
echo $cmd
done