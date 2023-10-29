data_dir="/data/datasets"
gpu=3
seeds=(0)
for seed in "${seeds[@]}";do
cmd="
python train.py --c config/usb_cv/cpmatch/cpmatch_cifar100_400_0.yaml 
--gpu=$gpu 
--seed=$seed
--save_name=cpmatch
--data_dir=$data_dir
--multiprocessing_distributed=False
--use_wandb
"
$cmd
echo $cmd
done