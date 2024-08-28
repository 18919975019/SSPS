#!/bin/bash
# log_file="train_SSPS_log/bce_for_sem.log"
# nohup python -u trainer_SSPS.py \
#                 --mode="train" \
#                 --segmentor_checkpoint="train_log/checkpoint_749.ckpt" \
#                 --eval_interval=500 \
#                 --log_dir="train_SSPS_log" \
#                 > "$log_file" 2>&1 &

log_file="train_SSPS_log/bce_for_sem_1.log"
nohup python -u trainer_SSPS.py \
                --mode="train" \
                --resume \
                --segmentor_checkpoint="train_SSPS_log/checkpoint_35.ckpt" \
                --eval_interval=500 \
                --log_dir="train_SSPS_log" \
                > "$log_file" 2>&1 &