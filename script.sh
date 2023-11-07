data_dir="/data/datasets/usb"
eval_batch_size=128
gpu=3
seeds=(0)
dataset="cifar100_300"
dataset="cifar100_400"
# dataset="cifar100_4000"
# dataset="tissuemnist_400"
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
confmatch_gamma=1.00
# cali_s="True"
cali_s="False"
conf_loss="True"
lambda_conf=0.001
# dataset="cifar100_300"
save_dir="./saved_models/usb_cv/${method}/"
for seed in "${seeds[@]}";do
# save_name="${method}_${dataset}_${seed}_${include_lb_to_ulb}_${confmatch_alpha}_${confmatch_delta}_${confmatch_gamma}_cali${n_repeat_loader_cali}"
# for FixMatch
save_name="${method}_${dataset}_${seed}_${include_lb_to_ulb}"
# for ConfMatch
save_name="${dataset}_${seed}_${include_lb_to_ulb}_${confmatch_alpha}_${confmatch_delta}_${confmatch_gamma}_cali${n_repeat_loader_cali}_conf${conf_loss}_${lambda_conf}"
load_path="--load_path=${save_dir}${save_name}/latest_model.pth"
cmd="
python train.py --c config/usb_cv/${method}/${method}_${dataset}_0.yaml 
--gpu=$gpu 
--seed=$seed
--eval_batch_size=$eval_batch_size
--save_name=$save_name
--data_dir=$data_dir
--multiprocessing_distributed=False
--n_repeat_loader_cali=$n_repeat_loader_cali
--confmatch_alpha=$confmatch_alpha
--confmatch_delta=$confmatch_delta
--confmatch_gamma=$confmatch_gamma
$use_wandb
--include_lb_to_ulb=$include_lb_to_ulb
--save_dir=$save_dir
--lambda_conf=$lambda_conf
--conf_loss $conf_loss
--confmatch_cali_s $cali_s
$load_path
"
$cmd
echo $cmd
done
# --conf_loss=$conf_loss

# saved_models/usb_cv/cpmatch_cifar100_400_0_True_0.1_0.1_1.00_cali5/latest_model.pth
# saved_models/usb_cv/cpmatch_cifar100_400_0_True_0.1_0.1_1.00_cali5/latest_model.pth