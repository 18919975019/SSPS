#!/bin/bash
# log_file="train_log_v3/bce_for_sem.log"
# nohup python -u trainer_base.py --mode="train" --eval_interval=100 --log_dir="train_log_v3" --resume --segmentor_checkpoint="train_log_v3/checkpoint_419.ckpt" > "$log_file" 2>&1 &

# pretrain in our way
# log_file="train_with_0.2_labeled/bce_for_sem_debug.log"
# nohup python -u trainer_base.py \
#                 --mode="train" \
#                 --batch_size='4,1' \
#                 --eval_interval=500 \
#                 --save_interval=5 \
#                 --log_dir="train_with_0.2_labeled" \
#                 --resume \
#                 --segmentor_checkpoint="train_with_0.2_labeled/checkpoint_879.ckpt" \
#                 > "$log_file" 2>&1 &

# pretrain in our way
# log_file="train_with_0.5_labeled/bce_for_sem_619.log"
# nohup python -u trainer_base.py \
#                 --mode="train" \
#                 --batch_size='4,1' \
#                 --eval_interval=20 \
#                 --save_interval=5 \
#                 --log_dir="train_with_0.5_labeled" \
#                 --resume \
#                 --segmentor_checkpoint="train_with_0.5_labeled/checkpoint_619.ckpt" \
#                 > "$log_file" 2>&1 &

# pretrain in mask3d way
# log_file="train_with_0.2_labeled_mask3d/bce_for_sem_14.log"
# nohup python -u trainer_base.py \
#                 --mode="train" \
#                 --batch_size='4,1' \
#                 --eval_interval=300 \
#                 --save_interval=5 \
#                 --log_dir="train_with_0.2_labeled_mask3d" \
#                 --resume \
#                 --segmentor_checkpoint="train_with_0.2_labeled_mask3d/checkpoint_14.ckpt" \
#                 > "$log_file" 2>&1 &

log_file="train_with_0.1_labeled_aug/bce_for_sem.log"
nohup python -u trainer_base.py \
                --mode="train" \
                --batch_size='4,1' \
                --eval_interval=1000 \
                --save_interval=5 \
                --log_dir="train_with_0.1_labeled_aug" \
                 --resume \
                --segmentor_checkpoint="train_with_0.1_labeled_aug/checkpoint_34.ckpt" \
                > "$log_file" 2>&1 &