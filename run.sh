
    
python main.py \
    --name "cifar10_lr0.05_p194_f50" \
    --lr 0.1 \
    --pretrain_ckpt "cifar10_lr0.05_complete/pretrain_e194" \
    --pass_pretrain

python main.py \
    --name "cifar100_lr0.05_p100_f50" \
    --lr 0.1 \
    --use_cifar100 \
    --pretrain_ckpt "cifar100_lr0.05_complete/cifar100_pretrain_e100" \
    --pass_pretrain

python main.py \
    --name "cifar100_lr0.05_p300_f50" \
    --lr 0.1 \
    --use_cifar100 \
    --pretrain_ckpt "cifar100_lr0.05_complete/cifar100_pretrain_e300" \
    --pass_pretrain
    
    
