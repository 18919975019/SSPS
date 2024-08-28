#!/bin/bash
# log_file="eval.log"
# nohup python -u trainer_base.py \
#                 --mode="eval" \
#                 --log_dir="eval_log" \
#                 --resume \
#                 --segmentor_checkpoint="train_log/checkpoint_775.ckpt" \
#                 > "$log_file" 2>&1 &
# log_file="eval_bce+dice.log"
# nohup python -u trainer_base.py --mode="eval" --log_dir="eval_log" --resume --segmentor_checkpoint="train_log_v2/checkpoint_133.ckpt" > "$log_file" 2>&1 &

# log_file="eval_pretrain_with_0.5_619.log"
# nohup python -u trainer_base.py \
#                 --mode="eval" \
#                 --log_dir="eval_log" \
#                 --resume \
#                 --segmentor_checkpoint="train_with_0.5_labeled/checkpoint_619.ckpt" \
#                 > "$log_file" 2>&1 &

# 起始 epoch
start_epoch=599
# 结束 epoch
end_epoch=724
# 递增步长
step=5

# 循环从 start_epoch 到 end_epoch，每次递增 step
for epoch in $(seq $start_epoch $step $end_epoch); do
    # 构造 checkpoint 文件名
    checkpoint_file="train_with_0.5_labeled/checkpoint_${epoch}.ckpt"
    # 构造日志文件名
    log_file="eval_pretrain_with_0.5_${epoch}.log"

    # 打印当前操作
    echo "Running for checkpoint: $checkpoint_file"

    # 使用 nohup 和 & 后台运行命令
    nohup python -u trainer_base.py \
                --mode="eval" \
                --log_dir="eval_log" \
                --resume \
                --segmentor_checkpoint="$checkpoint_file" \
                > "$log_file" 2>&1 &

    # 获取后台进程的PID
    pid=$!

    # 打印后台进程的PID
    echo "Started process for epoch $epoch with PID $pid"

    # 等待当前后台进程完成
    wait $pid

    # 打印当前进程完成信息
    echo "Process for epoch $epoch with PID $pid completed."

    # 可选：在每次运行后等待一段时间，避免紧接着启动下一个进程
    # sleep 5  # 取消注释此行来添加延迟

done

echo "All processes completed."
