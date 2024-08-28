import argparse
import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from omegaconf import OmegaConf
import hydra
import MinkowskiEngine as ME


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'Mask3D'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from Mask3D.third_party.pointnet2.pytorch_utils import BNMomentumScheduler
from Mask3D.models.mask3d import Mask3D
# from utils.tf_visualizer import Visualizer as TfVisualizer
# from models.loss_helper_labeled import get_labeled_loss
# from models.loss_helper_unlabeled import get_unlabeled_loss
# from models.loss_helper import compute_feature_consistency_loss, get_loss
from evaluator.PanopticEvaluator import PanopticEval
from datasets.make_dataloader import make_dataloader



def tb_name(key):
    if 'loss' in key:
        return 'loss/' + key
    elif 'acc' in key:
        return 'acc/' + key
    elif 'ratio' in key:
        return 'ratio/' + key
    elif 'value' in key:
        return 'value/' + key
    else:
        return 'other/' + key


def check_state_dict_consistency(loaded_state_dict, model_state_dict):
    """check consistency between loaded parameters and created model parameters
    """
    valid_state_dict = {}
    for k in loaded_state_dict:
        if k in model_state_dict:
            if loaded_state_dict[k].shape != model_state_dict[k].shape:
                print('\tSkip loading parameter {}, required shape{}, loaded shape{}'.format(
                          k, model_state_dict[k].shape, loaded_state_dict[k].shape))
                valid_state_dict[k] = model_state_dict[k]
            else:
                valid_state_dict[k] = loaded_state_dict[k]
        else:
            print('\tDrop parameter {}.'.format(k))

    for k in model_state_dict:
        if not (k in loaded_state_dict):
            print('\tNo param {}.'.format(k))
            valid_state_dict[k] = model_state_dict[k]

    return valid_state_dict

import yaml
def create_model(model_config_path, ema=False):
    # model = hydra.utils.instantiate(config)
    config = OmegaConf.load(model_config_path)
    # model = Mask3D(
    #     config=
    #     hidden_dim=128,
    #     num_queries=100,
    #     num_heads=8,
    #     dim_feedforward=1024,
    #     sample_sizes=[200, 800, 3200, 12800, 51200],
    #     shared_decoder=Truee,
    #     num_classes=20,
    #     num_decoders=3,
    #     dropout=0.,
    #     pre_norm=False,
    #     positional_encoding_type="fourier",
    #     non_parametric_queries=True,
    #     train_on_segments=True,
    #     normalize_pos_enc=True,
    #     use_level_embed=False,
    #     scatter_type="mean",
    #     hlevels=[0,1,2,3],
    #     use_np_features=False,
    #     voxel_size=0.02,
    #     max_sample_size=False,
    #     random_queries=False,
    #     gauss_scale=1.0,
    #     random_query_both=False,
    #     random_normal=False,
    #     dialations=[ 1, 1, 1, 1 ],
    #     conv1_kernel_size=5,
    #     bn_momentum=0.02,
    #     in_channels=3,
    #     out_channels=20,
    #     out_fpn=true
    # )

    model = hydra.utils.instantiate(config)
    
    # if ema:
    #     # for param in model.parameters():
    #     #     param.detach_()

    return model


class TrainerBase:
    def __init__(self, FLAGS, segmentor, ema_segmentor, device):
        # create log file
        self.LOG_DIR = FLAGS.log_dir
        if not os.path.exists(self.LOG_DIR):
            os.mkdir(self.LOG_DIR)
        self.LOG_FOUT = open(os.path.join(self.LOG_DIR, 'log_train.txt'), 'a')
        self.LOG_FOUT.write(str(FLAGS) + '\n')


        # 1.Create model
        self.segmentor = segmentor
        self.ema_segmentor = ema_segmentor
        for param in self.ema_segmentor.parameters():
            param.detach_()
        
        self.device = device
        # self.segmentor = create_model(segmentor_config).to(self.device)  # student model
        # self.ema_segmentor = create_model(segmentor_config, ema=True).to(self.device)  # teacher model
        
        # # if multi-gpu, parallelize model
        # if torch.cuda.device_count() > 1:
        #     self.log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
        #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #     self.segmentor = nn.parallel.DistributedDataParallel(self.segmentor, [0])
        #     self.ema_segmentor = nn.parallel.DistributedDataParallel(self.ema_segmentor, [0])

        # 2.Experiment setting
        batch_size_list = [int(x) for x in FLAGS.batch_size.split(',')]
        self.LABELED_BATCH_SIZE = batch_size_list[0]
        self.UNLABELED_BATCH_SIZE = batch_size_list[1]
        self.BATCH_SIZE = batch_size_list[0] + batch_size_list[1]  # 0 refers to labeled data, 1 refers to unlabeled data
        self.NUM_POINT = FLAGS.num_point
        self.MAX_EPOCH = FLAGS.max_epoch
        self.LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
        self.LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
        assert (len(self.LR_DECAY_STEPS) == len(self.LR_DECAY_RATES))
        # TFBoard Visualizers
        # self.TRAIN_VISUALIZER = TfVisualizer(self.LOG_DIR, 'train')
        # self.TEST_VISUALIZER = TfVisualizer(self.LOG_DIR, 'test')
        if not FLAGS.eval:
            self.PERFORMANCE_FOUT = open(os.path.join(self.LOG_DIR, 'best.txt'), 'w')
        self.BASE_LEARNING_RATE = FLAGS.learning_rate

        self.base_weight = FLAGS.base_weight  # weight for calculate consistency loss
        self.ema_decay = FLAGS.ema_decay
        self.print_interval = FLAGS.print_interval
        self.eval_interval = FLAGS.eval_interval
        self.save_interval = FLAGS.save_interval

        # 3.Load loss calculator and evaluator
        # self.train_labeled_criterion = get_labeled_loss
        # self.train_unlabeled_criterion = get_unlabeled_loss
        # self.test_detector_criterion = get_loss
        self.unlabeled_loss_weight = FLAGS.unlabeled_loss_weight
        self.evaluator = PanopticEval(20, ignore=255, min_points=1)

        # 4.Load the Adam optimizer and the BatchNorm scheduler
        self.make_optimizer(FLAGS.learning_rate, FLAGS.weight_decay)
        self.make_bnm_scheduler(FLAGS.bn_decay_step, FLAGS.bn_decay_rate)

        # 5. Load the dataloader
        self.make_dataloader()

        # 5.Load pretrained ckpt, resume from training[Optional]
        self.BEST_PQ = 0.
        self.is_resume = FLAGS.resume
        if FLAGS.segmentor_checkpoint is not None and os.path.isfile(FLAGS.segmentor_checkpoint):
            self.resume(FLAGS.segmentor_checkpoint, FLAGS.resume)

    def resume(self, ckpt, resume_training=False):
        # 1.get pretrained ckpt
        checkpoint = torch.load(ckpt)
        if "state_dict" in checkpoint.keys():
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint['model_state_dict']
        pretrained_dict = {k.replace("model.", ""): v for k, v in pretrained_dict.items()}

        # 2.check and load pretrained ckpt for student model and teacher model
        valid_state_dict = check_state_dict_consistency(pretrained_dict, self.segmentor.state_dict())
        self.segmentor.load_state_dict(valid_state_dict)
        valid_state_dict = check_state_dict_consistency(pretrained_dict, self.ema_segmentor.state_dict())
        self.ema_segmentor.load_state_dict(valid_state_dict)

        # 3.resume training
        if resume_training:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.BEST_PQ = checkpoint["BEST_PQ"]
            self.bnm_scheduler.step(self.start_epoch)
            print("-> Resume training from checkpoint %s (epoch: %d)" % (ckpt, self.start_epoch))
        else:
            print('-> Load pretrained checkpoint %s (epoch: %d)' % (ckpt, self.start_epoch))

    def make_optimizer(self, base_learning_rate, weight_decay):
        self.optimizer = optim.Adam(self.segmentor.parameters(), lr=base_learning_rate, weight_decay=weight_decay)

    def make_bnm_scheduler(self, bn_decay_step, bn_decay_rate):
        BN_MOMENTUM_INIT = 0.5
        BN_MOMENTUM_MAX = 0.001
        bn_lbmd = lambda it: max(
            BN_MOMENTUM_INIT * bn_decay_rate **
            (int(it / bn_decay_step)), BN_MOMENTUM_MAX)
        self.bnm_scheduler = BNMomentumScheduler(self.segmentor, bn_lambda=bn_lbmd, last_epoch=-1)

    def make_dataloader(self):
        self.LABELED_DATALOADER, self.UNLABELED_DATALOADER, self.TEST_DATALOADER = make_dataloader(
            self.LABELED_BATCH_SIZE, self.UNLABELED_BATCH_SIZE
        )

    def forward_segmentor(self, data, target, is_ema=False, is_eval=False):
        if len(data.coordinates) == 0:
            return None
        raw_coordinates = data.features[:, -3:]
        features = data.features[:, :-3]
        if raw_coordinates.shape[0] == 0:
            return None
        _data = ME.SparseTensor(coordinates=data.coordinates, features=features, device=self.device)
        point2segment = [pcd_target['point2segment'].to(self.device) for pcd_target in target]

       
        try:
            if is_ema:
                output = self.ema_segmentor(_data,
                                        point2segment=point2segment,
                                        raw_coordinates=raw_coordinates.to(self.device),
                                        is_eval=is_eval)
            else:
                output = self.segmentor(_data,
                                       point2segment=point2segment,
                                       raw_coordinates=raw_coordinates.to(self.device),
                                       is_eval=False)
                
        except RuntimeError as run_err:
            print(run_err)
            if 'only a single point gives nans in cross-attention' == run_err.args[0]:
                return None
            else:
                raise run_err
        return output



    def update_ema_variables(self, global_step):
        alpha = self.ema_decay
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(self.ema_segmentor.parameters(), self.segmentor.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def train_one_epoch(self, global_step, EPOCH_CNT):
        self.adjust_learning_rate(EPOCH_CNT)
        self.bnm_scheduler.step()  # decay BN momentum
        self.segmentor.train()  # set model to training mode
        self.ema_segmentor.train()
        
        unlabeled_dataloader_iterator = iter(self.UNLABELED_DATALOADER)
        for batch_idx, batch_data_target in enumerate(self.LABELED_DATALOADER):

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
            filename = filename = filename_unlabeled

            # 2.forward student model and teacher model separately
            self.optimizer.zero_grad()
            output = self.forward_segmentor(data, target, is_ema=False)
            with torch.no_grad():
                ema_output = self.forward_segmentor(data, target, is_ema=True)
            # 3.Compute loss and gradients, update parameters.
            loss = self.caculate_loss(output, ema_output, target)
            loss.backward()
            self.optimizer.step()
            global_step += 1
            self.update_ema_variables(global_step)

            # Accumulate statistics and print out
            end_points = {}
            end_points['loss'] = loss
            self.statistics(end_points)
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()


    def caculate_loss(self, output, ema_output, target):
        # TODO: need to implement the loss calculator module
        from loss.loss_labeled import get_labeled_loss, get_unlabeled_loss
        num_labeled = self.LABELED_BATCH_SIZE
        labeled_loss = get_labeled_loss(output, target, num_labeled, self.segmentor.module.train_on_segments)
        # labeled_loss, end_points = self.train_labeled_criterion(
        #     student_output, self.LABELED_BATCH_SIZE, DATASET_CONFIG, CONFIG_DICT)
        print(labeled_loss)
        unlabeled_loss = get_unlabeled_loss(output, ema_output, num_labeled, self.segmentor.module.train_on_segments)
        print(unlabeled_loss)
        exit()
        unlabeled_loss, end_points = self.train_unlabeled_criterion(
            student_output, teacher_output, self.LABELED_BATCH_SIZE, DATASET_CONFIG, CONFIG_DICT)

        features_consistency_loss = compute_feature_consistency_loss(
            end_points=student_output,
            ema_end_points=teacher_output,
            num_labeled=self.LABELED_BATCH_SIZE,
            cfg=CONFIG_DICT)
        consistency_weight = self.get_consistency_weight(EPOCH_CNT)
        loss = labeled_loss + unlabeled_loss * self.unlabeled_loss_weight + features_consistency_loss * consistency_weight
        return loss

    def statistics(self, end_points, batch_idx):
        stat_dict = {}  # collect statistics
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key or 'value' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()
        if (batch_idx + 1) % self.print_interval == 0:
            self.log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            self.TRAIN_VISUALIZER.log_scalars(
                {
                    tb_name(key): stat_dict[key] / self.print_interval
                    for key in stat_dict
                },
                (EPOCH_CNT * len(self.LABELED_DATALOADER) + batch_idx) * self.BATCH_SIZE)
            for key in sorted(stat_dict.keys()):
                self.log_string('mean %s: %f' %
                                (key, stat_dict[key] / self.print_interval))
                stat_dict[key] = 0

    def evaluate_one_epoch(self):
        stat_dict = {}  # collect statistics
        self.segmentor.eval()  # set model to eval mode (for bn and dp)
        for batch_idx, batch_data_target in enumerate(self.TEST_DATALOADER):
            # 1.forward frozen student model
            data, target, filename = batch_data_target
            with torch.no_grad():
                output = self.forward_segmentor(data, target, is_ema=False, is_eval=True)

            # 2.compute batch PQ
            loss, end_points = self.test_criterion(output, DATASET_CONFIG)
            # Accumulate statistics and print out
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'ratio' in key or 'value' in key:
                    if key not in stat_dict: stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

            sem_pred, inst_pred, sem_gt, inst_gt = self.parse_output(output)
            # TODO: postprocess to get sem_pred, inst_pred, sem_gt, inst_gt.Must be batched point-wise output like [B,N]
            self.evaluator.addBatch(sem_pred, inst_pred, sem_gt, inst_gt)

        # Log statistics
        for key in sorted(stat_dict.keys()):
            self.log_string('eval mean %s: %f' % (key, stat_dict[key] /
                                             (float(batch_idx + 1))))
        # TODO: stat_dict should include key 'segmentation_loss'
        mean_loss = stat_dict['segmentation_loss'] / float(batch_idx + 1)

        pq, sq, rq, all_pq, all_sq, all_rq = self.evaluator.getPQ()
        return mean_loss, pq

    def parse_output(self, output):
        pass
        return sem_pred, inst_pred, sem_gt, inst_gt

    def train(self):
        # Training
        start_from = 0
        global_step = 0
        loss = 0
        if self.is_resume:
            start_from = self.start_epoch
        for epoch in range(start_from, self.MAX_EPOCH):
            self.log_string('\n**** EPOCH %03d, STEP %d ****' % (epoch, global_step))
            self.log_string('Current learning rate: %f' % (self.get_current_lr(epoch)))
            self.log_string('Current BN decay momentum: %f' % (self.bnm_scheduler.lmbd(self.bnm_scheduler.last_epoch)))
            self.log_string(str(datetime.now()))
            np.random.seed()
            global_step = self.train_one_epoch(global_step, epoch)

            # evaluate
            if epoch > 0 and epoch % self.eval_interval == 0:
                loss, pq = self.evaluate_one_epoch()
                if pq > self.BEST_PQ:
                    self.BEST_PQ = pq
                    self.save_model(epoch, loss)
                self.PERFORMANCE_FOUT.write('epoch: ' + str(epoch) + '\n' + 'best: ' + str(self.BEST_PQ.item()) + '\n')
                self.PERFORMANCE_FOUT.flush()

            # save model
            if epoch % self.save_interval == 0:
                self.save_model(epoch, loss)

    def pretrain_calculate_loss(self, output, target):
        from loss.loss_labeled import get_loss
        labeled_loss = get_loss(output, target, self.segmentor.module.train_on_segments)
        return labeled_loss

    def pretrain_one_epoch(self, global_step, EPOCH_CNT):
        self.adjust_learning_rate(EPOCH_CNT)
        self.bnm_scheduler.step()  # decay BN momentum
        self.segmentor.train()  # set model to training mode
        for batch_idx, batch_data_target in enumerate(self.LABELED_DATALOADER):

            # 1.forward student model
            data, target, filename = batch_data_target
            self.optimizer.zero_grad()
            output = self.forward_segmentor(data, target, is_ema=False)
            # 2.compute loss and gradients, update parameters.
            loss = self.pretrain_calculate_loss(output, target)
            loss.backward()
            self.optimizer.step()
            global_step += 1
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        return global_step

    def pretrain(self):
        
        start_from = 0
        global_step = 0
        loss = 0
        if self.is_resume:
            start_from = self.start_epoch
        for epoch in range(start_from, self.MAX_EPOCH):
            # Training
            self.log_string('\n**** EPOCH %03d, STEP %d ****' % (epoch, global_step))
            self.log_string('Current learning rate: %f' % (self.get_current_lr(epoch)))
            self.log_string('Current BN decay momentum: %f' % (self.bnm_scheduler.lmbd(self.bnm_scheduler.last_epoch)))
            self.log_string(str(datetime.now()))
            np.random.seed()
            global_step = self.pretrain_one_epoch(global_step, epoch)
            # evaluate
            if epoch > 0 and epoch % self.eval_interval == 0:
                loss, pq = self.evaluate_one_epoch()
                if pq > self.BEST_PQ:
                    self.BEST_PQ = pq
                    self.save_model(epoch, loss)
                self.PERFORMANCE_FOUT.write('epoch: ' + str(epoch) + '\n' + \
                                            'best: ' + str(self.BEST_PQ.item()) + '\n')
                self.PERFORMANCE_FOUT.flush()
            # save model
            if epoch % self.save_interval == 0:
                self.save_model(epoch, loss)

    def save_model(self, epoch, loss):
        save_dict = {
            'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,  # loss during evaluation,
            'BEST_PQ': self.BEST_PQ
        }
        try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = self.segmentor.module.state_dict()
            save_dict[
                'ema_model_state_dict'] = self.ema_segmentor.module.state_dict()
        except:
            save_dict['model_state_dict'] = self.segmentor.state_dict()
            save_dict['ema_model_state_dict'] = self.ema_segmentor.state_dict()
        torch.save(save_dict, os.path.join(self.LOG_DIR, 'checkpoint_%d.tar' % epoch))

    def log_string(self, out_str):
        self.LOG_FOUT.write(out_str + '\n')
        self.LOG_FOUT.flush()
        print(out_str)

    def get_consistency_weight(self, epoch):
        base_weight = self.base_weight
        for i, lr_decay_epoch in enumerate(self.LR_DECAY_STEPS):
            if epoch >= lr_decay_epoch:
                base_weight *= 2
        return base_weight

    def get_current_lr(self, epoch):
        # stairstep update
        lr = self.BASE_LEARNING_RATE
        for i, lr_decay_epoch in enumerate(self.LR_DECAY_STEPS):
            if epoch >= lr_decay_epoch:
                lr *= self.LR_DECAY_RATES[i]
        return lr

    def adjust_learning_rate(self, epoch):
        lr = self.get_current_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                    default='2,4',
                    help='Batch Size during training, labeled + unlabeled')
parser.add_argument('--num_point',
                    type=int,
                    default=40000,
                    help='Point Number')
parser.add_argument('--max_epoch', type=int, default=1001, help='Epoch to run')
parser.add_argument('--lr_decay_steps',
                    default='400, 600, 800, 900',
                    help='When to decay the learning rate (in epochs)')
parser.add_argument('--lr_decay_rates',
                    default='0.3, 0.3, 0.1, 0.1',
                    help='Decay rates for lr decay')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.002,
                    help='Initial learning rate')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0,
                    help='Optimization L2 weight decay')
parser.add_argument('--bn_decay_step',
                    type=int,
                    default=20,
                    help='Period of BN decay (in epochs)')
parser.add_argument('--bn_decay_rate',
                    type=float,
                    default=0.5,
                    help='Decay rate for BN decay')
parser.add_argument('--log_dir',
                    default='temp',
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
                    default=25,
                    help='batch interval to print loss')
parser.add_argument('--eval_interval',
                    type=int,
                    default=25,
                    help='epoch interval to evaluate model')
parser.add_argument('--save_interval',
                    type=int,
                    default=200,
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

import torch.multiprocessing as mp
import torch.distributed as dist
def main():
    
    # segmentor_config_path = "/home/zxhong/SSPS/Mask3D/conf/model/segmentor.yaml"
    # trainer = TrainerBase(FLAGS, segmentor_config_path)
    # trainer.train()
    # loss and network
    FLAGS = parser.parse_args()
    num_devices = torch.cuda.device_count()-1
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
 
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    ema_model = torch.nn.parallel.DistributedDataParallel(ema_model, device_ids=[args.gpu])

    trainer = TrainerBase(args, segmentor=model, ema_segmentor=ema_model, device=args.gpu)
    trainer.pretrain()
    # trainer.train()


if __name__ == '__main__':
    main()
    