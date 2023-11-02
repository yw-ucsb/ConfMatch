data_dir="/data/datasets"
gpu=4
seeds=(0)
dataset="cifar100_400"
method="cpmatch"
method="fixmatch"
dataset="cifar100_300"
for seed in "${seeds[@]}";do
cmd="
python train.py --c config/usb_cv/${method}/${method}_${dataset}_0.yaml 
--gpu=$gpu 
--seed=$seed
--save_name=${method}_${dataset}_${seed}
--data_dir=$data_dir
--multiprocessing_distributed=False
--use_wandb
"
$cmd
echo $cmd
done