import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import hydra
import MinkowskiEngine as ME
from torch_scatter import scatter_mean
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import wandb
import gc
from trainer_base import TrainerBase

def create_model(model_config_path, ema=False):
    config = OmegaConf.load(model_config_path)
    model = hydra.utils.instantiate(config)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        default='4,4',
                        help='Batch Size during training, labeled + unlabeled')
    parser.add_argument('--num_point',
                        type=int,
                        default=40000,
                        help='Point Number')
    parser.add_argument('--max_epoch', type=int, default=1000, help='Epoch to run')
    parser.add_argument('--lr_decay_steps',
                        default='400, 600, 800, 900',
                        help='When to decay the learning rate (in epochs)')
    parser.add_argument('--lr_decay_rates',
                        default='0.3, 0.3, 0.1, 0.1',
                        help='Decay rates for lr decay')
    parser.add_argument('--log_dir',
                        default='log',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--base_weight', type=float, default=0.001)
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        metavar='ALPHA',
                        help='ema variable decay rate')
    parser.add_argument('--print_interval',
                        type=int,
                        default=2,
                        help='batch interval to print loss')
    parser.add_argument('--eval_interval',
                        type=int,
                        default=10,
                        help='epoch interval to evaluate model')
    parser.add_argument('--save_interval',
                        type=int,
                        default=2,
                        help='epoch interval to save model')
    parser.add_argument('--unlabeled_loss_weight',
                        type=float,
                        default=2.0,
                        metavar='WEIGHT',
                        help='use unlabeled loss with given weight')
    parser.add_argument('--segmentor_checkpoint',
                        default=None,
                        help='the pre-trained checkpoint for segmentor')
    parser.add_argument('--resume',
                        action='store_true',
                        help='resume training instead of just loading a pre-train model')
    parser.add_argument('--mode',
                        default="train",
                        help='decide train or eval')
    FLAGS = parser.parse_args()
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    num_devices = 1
    print(
        "Testing ",
        num_devices,
        " GPUs. Total batch size: ",
        num_devices * FLAGS.batch_size,
    )
    FLAGS.world_size = num_devices
    mp.spawn(main_worker, nprocs=num_devices, args=(num_devices, FLAGS))
    

def main_worker(gpu, ngpus_per_node, args):
    global min_time
    if args.world_size == 1:
        args.gpu = 0
    else:
        args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    args.rank = 0 * ngpus_per_node + gpu
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:34567",
        world_size=args.world_size,
        rank=args.rank,
    )
    segmentor_config_path = "/home/zxhong/SSPS/Mask3D/conf/model/segmentor.yaml"
    model = create_model(segmentor_config_path)
    ema_model = create_model(segmentor_config_path, ema=True)
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    ema_model.cuda(args.gpu)
 
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    ema_model = torch.nn.parallel.DistributedDataParallel(ema_model, device_ids=[args.gpu])

    postprocess_config_path = "postprocess.yaml"
    postprocess_config = OmegaConf.load(postprocess_config_path)
    optimize_config_path = "optimize.yaml"
    optimize_config = OmegaConf.load(optimize_config_path)
    test_config_path = "test_config.yaml"
    test_config = OmegaConf.load(test_config_path)
    trainer = TrainerBase(args, 
                          segmentor=model, 
                          ema_segmentor=ema_model, 
                          device=args.gpu, 
                          config=postprocess_config, 
                          optimize_config=optimize_config, 
                          test_config=test_config)
    
    import datetime
    current_time = datetime.datetime.now()
    experiment_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"pretrain backbone at {experiment_name}"
    wandb.init(project="SSPanoptic Segmentation", config=vars(args), name=experiment_name)
    if args.mode == "pretrain":
        trainer.pretrain()
    elif args.mode == "eval":
        trainer.evaluate_one_epoch()
    elif args.mode == "train":
        trainer.train()
    wandb.finish()
    


if __name__ == '__main__':
    main()