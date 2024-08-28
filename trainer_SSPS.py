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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'Mask3D'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from Mask3D.third_party.pointnet2.pytorch_utils import BNMomentumScheduler
from evaluator.PanopticEvaluator import PanopticEval
from evaluator.MapCreator import MapCreator
from datasets.make_dataloader import make_dataloader
CLASS_DICT = {
    0: 'wall',
    1: 'floor',
    2: 'cabinet',
    3: 'bed',
    4: 'chair',
    5: 'sofa',
    6: 'table',
    7: 'door',
    8: 'window',
    9: 'bookshelf',
    10: 'picture',
    11: 'counter',
    12: 'desk',
    13: 'curtain',
    14: 'refrigerator',
    15: 'shower curtain',
    16: 'toilet',
    17: 'sink',
    18: 'bathtub',
    19: 'other furniture',
}

from trainer_base import TrainerBase
class TrainerSSPS(TrainerBase):
    def __init__(self, FLAGS, segmentor, ema_segmentor, device, config, optimize_config, test_config):
        super().__init__(FLAGS, segmentor, ema_segmentor, device, config, optimize_config, test_config)
    
    def update_ema_variables(self, global_step):
        alpha = self.ema_decay
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(self.ema_segmentor.parameters(), self.segmentor.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def train(self):
        start_from = 0
        global_step = 0
        loss = 0
        if self.is_resume:
            start_from = self.start_epoch
        for epoch in range(start_from, self.MAX_EPOCH):
            # Train
            self.log_string('\n**** EPOCH %03d, STEP %d ****' % (epoch, global_step))
            self.log_string('Current learning rate: %f' % (self.optimizer.param_groups[0]['lr']))
            self.log_string('Current BN decay momentum: %f' % (self.bnm_scheduler.lmbd(self.bnm_scheduler.last_epoch)))
            self.log_string(str(datetime.now()))
            np.random.seed()
            global_step = self.train_one_epoch(global_step, epoch)
            # Save model
            if (epoch+1) % self.save_interval == 0:
                self.save_model(epoch, loss)
            # Evaluate
            if epoch > 0 and (epoch+1) % self.eval_interval == 0:
                pq, loss = self.evaluate_one_epoch()
                if pq > self.BEST_PQ:
                    self.BEST_PQ = pq
                    self.save_model(epoch, loss)
                self.PERFORMANCE_FOUT.write('epoch: ' + str(epoch) + '\n' + \
                                            'best: ' + str(self.BEST_PQ.item()) + '\n')
                self.PERFORMANCE_FOUT.flush()
            
    def train_one_epoch(self, global_step, epoch):
        self.adjust_learning_rate(epoch)
        self.bnm_scheduler.step()  # decay BN momentum
        self.segmentor.train()  # set model to training mode
        self.ema_segmentor.train()
        losses_dict = {
            "loss_ce": [],
            "loss_mask": [],
            "loss_dice": [],
            "semantic_loss_mask": [],
            "unsupervise_loss_ce": [],
            "unsupervise_loss_mask": [],
            "unsupervise_loss_dice": [],
            "unsupervise_semantic_loss_mask": [],
            # "semantic_loss_ce": [],
            # "semantic_loss_dice": [],
        }
        losses = []

        t_bar = tqdm(self.LABELED_DATALOADER)
        unlabeled_dataloader_iterator = iter(self.UNLABELED_DATALOADER)
        for batch_idx, batch_data_target in enumerate(t_bar):

            # 1.concat labeled batch data and unlabeled data
            data, target, filename = batch_data_target
            try:
                data_target_unlabeled = next(unlabeled_dataloader_iterator)
            except StopIteration:
                unlabeled_dataloader_iterator = iter(self.UNLABELED_DATALOADER)
                data_target_unlabeled = next(unlabeled_dataloader_iterator)
            data_unlabeled, target_unlabeled, filename_unlabeled = data_target_unlabeled

            data = data + data_unlabeled
            target =  target + target_unlabeled
            filename = filename + filename_unlabeled
            
            # 2.forward student model and teacher model separately
            self.optimizer.zero_grad()
            output = self.forward_segmentor(data, target, is_ema=False)
            with torch.no_grad():
                ema_output = self.forward_segmentor(data, target, is_ema=True)
            
            # 3.Compute loss and gradients, update parameters.
            loss, loss_dict = self.calculate_loss(output, ema_output, target, data)
            loss.backward()
            self.optimizer.step()
            global_step += 1
            self.update_ema_variables(global_step)
            # Accumulate statistics and print out
            losses.append(loss.detach().cpu().item())
            mean_loss = sum(losses)/len(losses)
            wandb.log({"train_mean_loss":mean_loss})
            t_bar.set_postfix(epoch=f"{epoch}", mean_loss=f'{mean_loss:.4f}')
            for k,v in loss_dict.items():
                if k in losses_dict.keys():
                    losses_dict[k].append(v.detach().cpu().item())
                    mean = sum(losses_dict[k])/len(losses_dict[k])
                    wandb.log({k:mean})
            del output
            del data
            gc.collect()
            torch.cuda.empty_cache()
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        # self.lr_scheduler.step()
        return global_step
    
    def calculate_loss(self, output, ema_output, target, data):
        loss_dict = {}
        loss_weight_dict = {
            "supervise_loss": 1.,
            "unsupervise_loss": 1.
        }
        from loss.loss_labeled import get_labeled_loss, get_unlabeled_loss
        labeled_loss, labeled_loss_dict = get_labeled_loss(output,
                                                           target, 
                                                           self.LABELED_BATCH_SIZE, 
                                                           self.segmentor.module.train_on_segments)
        unlabeled_loss, unlabeled_loss_dict  = get_unlabeled_loss(output, 
                                                                  ema_output, 
                                                                  self.LABELED_BATCH_SIZE, 
                                                                  self.segmentor.module.train_on_segments,
                                                                  data,
                                                                  target,
                                                                  self.map_creator)
        loss_dict.update(labeled_loss_dict)
        for loss in unlabeled_loss_dict:
            loss_dict[f"unsupervise_"+loss] = unlabeled_loss_dict[loss]
        
        for loss in loss_dict.keys():
            if "unsupervise" in loss:
                loss_dict[loss] *= loss_weight_dict["unsupervise_loss"]
            else:
                loss_dict[loss] *= loss_weight_dict["supervise_loss"]
        return sum(loss_dict.values()), loss_dict
    
def create_model(model_config_path, ema=False):
    config = OmegaConf.load(model_config_path)
    model = hydra.utils.instantiate(config)
    return model

import torch.multiprocessing as mp
import torch.distributed as dist
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        default='2,2',
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
                        default="train_log/checkpoint_749.ckpt",
                        help='the pre-trained checkpoint for segmentor')
    parser.add_argument('--resume',
                        action='store_true',
                        help='resume training instead of just loading a pre-train model')
    parser.add_argument('--mode',
                        default="train",
                        help='decide train or eval')
    parser.add_argument('--device',
                        default="1",
                        help="which gpu is chosen for experiment")
    FLAGS = parser.parse_args()
    
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device
    ngpus_per_node = int((len(FLAGS.device)+1)/2)
    FLAGS.world_size = ngpus_per_node
    print(
        "Testing ",
        ngpus_per_node,
        " GPUs. Total batch size: ",
        ngpus_per_node * FLAGS.batch_size,
    )
    # for 1 node, assign 1 procs for 1 gpu, nprocs=ngpus_per_node
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, FLAGS)) 
    

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
        init_method="tcp://localhost:23456",
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
    trainer = TrainerSSPS(args, segmentor=model, 
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
    if args.mode == "train":
        trainer.train()
    elif args.mode == "eval":
        trainer.evaluate_one_epoch()
    # trainer.train()
    wandb.finish()
    


if __name__ == '__main__':
    main()