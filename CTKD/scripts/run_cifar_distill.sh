# Use this script to train your own student model.

# PKT
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet20 -r 0.1 -a 0.9 -b 30000
# SP
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill similarity --model_s resnet20 -r 0.1 -a 0.9 -b 3000
# VID
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill vid --model_s resnet20 -r 0.1 -a 0.9 -b 1
# CRD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet20 -r 0.1 -a 0.9 -b 0.8
# SRRL
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill srrl --model_s resnet20 -r 0.1 -a 0.9 -b 1
# DKD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill dkd --model_s resnet20 -r 1 -a 0 -b 1 --dkd_alpha 1 --dkd_beta 2


PYTHONPATH="/data/mmc_lyxiang/KD/logit-standardization-KD-master/" python3 train_student.py --model_t resnet32x4 \
        --distill ekd \
        --model_s resnet8x4 -r 1.0 -a 4.5 -b 0 --kd_T 4 \
        --batch_size 64 --learning_rate 0.05 \
        --have_mlp 1 --mlp_name 'global'\
        --cosine_decay 1 --decay_max 0 --decay_min -1 --decay_loops 10 \
        --save_model \
        --experiments_dir 'tea-res32x4-stu-res8x4/ekd/global_T' \
        --experiments_name 'setting63_CTKDonly,T=4' \
        --deterministic \
        --lamb -1.22 \
        --wandb \
        --project Resnet32x4_Resnet8x4_CTKD \
        --gpu_id 3 \
        --warmup 30
        
