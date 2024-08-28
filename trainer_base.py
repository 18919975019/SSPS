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

def create_model(model_config_path, ema=False):
    config = OmegaConf.load(model_config_path)
    model = hydra.utils.instantiate(config)
    return model

def get_mask_and_scores(pred_logits, pred_masks, device, num_queries=80, num_classes=18, topk_per_scene=-1):
    """
    :param pred_logits: predicted logits for 1 scene, (n_masks x n_classes)
    :param pred_masks: predicted masks for 1 scene, (n_voxels x n_masks)
    :return:
    """
    labels = (
        torch.arange(num_classes, device=device)
        .unsqueeze(0)
        .repeat(num_queries, 1)
        .flatten(0, 1)
    )
    cls_scores = F.softmax(pred_logits, dim=-1)[:, :-1]
    if topk_per_scene != -1:
        topk_cls_scores, topk_indices = cls_scores.flatten(0, 1).topk(
            topk_per_scene, sorted=True
        )
    else:
        topk_cls_scores, topk_indices = cls_scores.flatten(0, 1).topk(
            num_queries, sorted=True
        )

    topk_labels = labels[topk_indices]
    topk_indices = topk_indices // num_classes
    pred_masks = pred_masks[:, topk_indices]
    heatmap = pred_masks.float().sigmoid()
    pred_binary_masks = (pred_masks > 0).float()
    topk_mask_scores = (heatmap * pred_binary_masks).sum(0) / (pred_binary_masks.sum(0) + 1e-6)
    scores = topk_cls_scores * topk_mask_scores

    # sort masks, heatmap, labels witn scores
    sort_scores = scores.sort(descending=True)
    sort_scores_index = sort_scores.indices.cpu()
    sort_scores_values = sort_scores.values.cpu()
    sorted_masks = pred_binary_masks[:, sort_scores_index]
    sorted_heatmap = heatmap[:, sort_scores_index]
    sort_labels = topk_labels[sort_scores_index]

    # filter out pred for stuff and uninterested
    instance_pred_masking = (sort_labels!=1) & (sort_labels!=0) & (sort_labels!=20)
    sorted_masks = sorted_masks[:,instance_pred_masking]
    sorted_heatmap = sorted_heatmap[:,instance_pred_masking]
    sort_labels = sort_labels[instance_pred_masking]
    sort_scores_values = sort_scores_values[instance_pred_masking]


    return sorted_masks, sorted_heatmap, sort_labels, sort_scores_values

def get_full_res_mask(mask, inverse_map, point2segment_full, eval_on_segments=True, is_heatmap=False):
        mask = mask.detach().cpu()[inverse_map]  # full res points
        # scatter binary predictions on segment and map back to points, points->segemnts->points
        if eval_on_segments and is_heatmap == False:
            mask = scatter_mean(mask, point2segment_full, dim=0)  # full res segments
            mask = (mask > 0.5).float()
            mask = mask.detach().cpu()[point2segment_full.cpu()]  # full res points

        return mask

def NMS(sorted_masks, sorted_heatmap, sort_classes, sort_scores_values, scores_threshold=0.3, iou_threshold=0.5):
    all_pred_classes = list()
    all_pred_masks = list()
    all_pred_scores = list()
    all_heatmaps = list()
    keep_instances = set()
    pairwise_overlap = sorted_masks.T @ sorted_masks
    normalization = pairwise_overlap.max(axis=0)
    norm_overlaps = pairwise_overlap / normalization
    for instance_id in range(norm_overlaps.shape[0]):
        # filter out unlikely masks and nearly empty masks
        # if not(sort_scores_values[instance_id] < 0.3 or sorted_masks[:, instance_id].sum() < 500):
        if not (
                sort_scores_values[instance_id]
                < scores_threshold
        ):
            # check if mask != empty
            if not sorted_masks[:, instance_id].sum() == 0.0:
                overlap_ids = set(
                    np.nonzero(
                        norm_overlaps[instance_id, :]
                        > iou_threshold
                    )[0]
                )
                if len(overlap_ids) == 0:
                    keep_instances.add(instance_id)
                else:
                    if instance_id == min(overlap_ids):
                        keep_instances.add(instance_id)

    keep_instances = sorted(list(keep_instances))
    return sort_classes[keep_instances], sorted_masks[:, keep_instances], sort_scores_values[keep_instances], sorted_heatmap[:, keep_instances]

def create_map(semantic_masks, semantic_heatmaps, instance_masks, instance_labels, instance_scores):
    """
    semanitc_masks:(n_points x n_classes)
    semantic_labels:labels for each semanitc masks, (n_classes)
    semantic_scores: score for each semanitc masks, (n_classes)
    instance_masks:(n_points x n_masks)
    instance_labels:labels for each instance masks, (n_masks)
    instance_scores: score for each instance masks, (n_masks)

    return: semantic_map: (n_points), assign each point a semantic label
            instance_map: (n_points), assign each point a instance label
    """
    num_points = semantic_masks.shape[0]
    num_classes = semantic_masks.shape[1]
    # calculate semantic map, need to add mask for void area, label void area as 20
    semantic_masks_scores = (semantic_heatmaps * semantic_masks).sum(0) / (semantic_masks.sum(0) + 1e-6)
    uninterested_semantic_mask = np.zeros((num_points,1))
    uninterested_semantic_score = [0.]
    _semantic_masks = np.concatenate([uninterested_semantic_mask, semantic_masks], axis=1)
    _semantic_scores = np.concatenate([uninterested_semantic_score, semantic_masks_scores])
    semantic_map = np.argmax(_semantic_masks * _semantic_scores[None, :], axis=1)
    semantic_map = semantic_map - 1
    semantic_map[semantic_map==-1]=num_classes
    # calculate instance map, need to add mask for void area, label void area as -1
    uninterested_instance_mask = np.zeros((num_points,1))
    uninterested_instance_score = [0.]
    _instance_masks = np.concatenate([uninterested_instance_mask, instance_masks], axis=1)
    _instance_scores = np.concatenate([uninterested_instance_score, instance_scores])
    instance_map = np.argmax(_instance_masks * _instance_scores[None, :], axis=1)
    instance_map = instance_map - 1
    instance_semantic_map = torch.full(instance_map.shape, 20)
    for i in range(len(instance_labels)):
        instance_semantic_map[instance_map == i] == instance_labels[i]
    print(instance_semantic_map)
    print(semantic_map)
    return semantic_map, instance_map

def check_state_dict_consistency(loaded_state_dict, model_state_dict):
    """
    check consistency between loaded parameters and created model parameters
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

class TrainerBase:
    def __init__(self, FLAGS, segmentor, ema_segmentor, device, config, optimize_config, test_config):
        # 1.create log file
        self.LOG_DIR = FLAGS.log_dir
        if not os.path.exists(self.LOG_DIR):
            os.mkdir(self.LOG_DIR)
        self.LOG_FOUT = open(os.path.join(self.LOG_DIR, 'log_train.txt'), 'a')
        self.LOG_FOUT.write(str(FLAGS) + '\n')
        # 2.create student model and teacher model, detach the parameters for teacher model
        self.segmentor = segmentor
        self.ema_segmentor = ema_segmentor
        for param in self.ema_segmentor.parameters():
            param.detach_()
        self.device = device
        # 3.experiment hyperparameters setting
        self.test_config = test_config
        batch_size_list = [int(x) for x in FLAGS.batch_size.split(',')]
        self.LABELED_BATCH_SIZE = batch_size_list[0]
        self.UNLABELED_BATCH_SIZE = batch_size_list[1]
        self.BATCH_SIZE = batch_size_list[0] + batch_size_list[1]  # 0 refers to labeled data, 1 refers to unlabeled data
        self.NUM_POINT = FLAGS.num_point
        self.MAX_EPOCH = FLAGS.max_epoch
        self.LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
        self.LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
        assert (len(self.LR_DECAY_STEPS) == len(self.LR_DECAY_RATES))
        if not FLAGS.eval:
            self.PERFORMANCE_FOUT = open(os.path.join(self.LOG_DIR, 'best.txt'), 'w')
        self.BASE_LEARNING_RATE = optimize_config.base_learning_rate
        self.base_weight = FLAGS.base_weight  # weight for calculate consistency loss
        self.ema_decay = FLAGS.ema_decay
        self.print_interval = FLAGS.print_interval
        self.eval_interval = FLAGS.eval_interval
        self.save_interval = FLAGS.save_interval
        # 4.Load the dataloader
        self.make_dataloader()
        # 5.Load evaluator
        self.unlabeled_loss_weight = FLAGS.unlabeled_loss_weight
        self.evaluator = PanopticEval(21, ignore=[20], min_points=1)
        self.map_creator = MapCreator(test_config, self.device)
        # 6.Load the Adam optimizer, lr scheduler and the BatchNorm scheduler
        self.make_optimizer_and_scheduler(optimize_config)
        # 7.Load pretrained ckpt, resume from training[Optional]
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
        pretrained_dict = {f"module.{k}": v for k, v in pretrained_dict.items()}
        # pretrained_dict = {k.replace("model.", ""): v for k, v in pretrained_dict.items()}

        # 2.check and load pretrained ckpt for student model and teacher model
        valid_state_dict = check_state_dict_consistency(pretrained_dict, self.segmentor.state_dict())
        self.segmentor.load_state_dict(valid_state_dict)
        valid_state_dict = check_state_dict_consistency(pretrained_dict, self.ema_segmentor.state_dict())
        self.ema_segmentor.load_state_dict(valid_state_dict)

        # 3.resume training
        if resume_training:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.BEST_PQ = checkpoint["BEST_PQ"]
            self.bnm_scheduler.step(self.start_epoch)
            print("-> Resume training from checkpoint %s (epoch: %d)" % (ckpt, self.start_epoch))
        else:
            print('-> Load pretrained checkpoint %s' % (ckpt))

    def make_dataloader(self):
        self.LABELED_DATALOADER, self.UNLABELED_DATALOADER, self.TEST_DATALOADER = make_dataloader(
            self.LABELED_BATCH_SIZE, self.UNLABELED_BATCH_SIZE
        )

    def make_optimizer_and_scheduler(self, optimize_config):
        self.optimizer = optim.AdamW(
            params=self.segmentor.parameters(), 
            lr=optimize_config.base_learning_rate,
            weight_decay=optimize_config.weight_decay
            )
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=optimize_config.base_learning_rate,
            epochs=self.MAX_EPOCH,
            steps_per_epoch=len(self.LABELED_DATALOADER),
            )
        BN_MOMENTUM_INIT = 0.5
        BN_MOMENTUM_MAX = 0.001
        bn_lbmd = lambda it: max(
            BN_MOMENTUM_INIT * optimize_config.bn_decay_rate **(int(it / optimize_config.bn_decay_step)), 
            BN_MOMENTUM_MAX)
        self.bnm_scheduler = BNMomentumScheduler(
            self.segmentor, 
            bn_lambda=bn_lbmd, 
            last_epoch=-1
            )

    def save_model(self, epoch, loss):
        save_dict = {
            'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
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
        torch.save(save_dict, os.path.join(self.LOG_DIR, 'checkpoint_%d.ckpt' % epoch))

    def log_string(self, out_str):
        self.LOG_FOUT.write(out_str + '\n')
        self.LOG_FOUT.flush()
        print(out_str)

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

    def forward_segmentor(self, data, target, is_ema=False, is_eval=False):
        if is_ema:
            if len(data.ema_coordinates) == 0:
                return None
            ema_raw_coordinates = data.ema_features[:, -3:]
            ema_features = data.ema_features[:, :-3]
            if ema_raw_coordinates.shape[0] == 0:
                return None
            ema_data = ME.SparseTensor(coordinates=data.ema_coordinates, features=ema_features, device=self.device)
            ema_point2segment = [pcd_target['ema_point2segment'].to(self.device) for pcd_target in target]     
        else:
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
                output = self.ema_segmentor(ema_data,
                                        point2segment=ema_point2segment,
                                        raw_coordinates=ema_raw_coordinates.to(self.device),
                                        is_eval=False)
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
        
        if is_eval:
            return output, raw_coordinates
        return output
    
    def pretrain_calculate_loss(self, output, target):
        from loss.loss_labeled import get_segmentation_loss
        loss, loss_dict = get_segmentation_loss(output, target, self.segmentor.module.train_on_segments)

        # from loss.loss_labeled import get_mask3d_segmentation_loss
        # for tgt in target:
        #     tgt["masks"] = tgt["masks"][:-18]
        #     tgt["labels"] = tgt["labels"][:-18]
        #     tgt["segment_mask"] = tgt["segment_mask"][:-18]
        # loss, loss_dict = get_mask3d_segmentation_loss(output, target, self.segmentor.module.train_on_segments)

        # from loss.loss_labeled import get_thing_stuff_loss
        # loss, loss_dict = get_thing_stuff_loss(output, target, self.segmentor.module.train_on_segments)
        return loss, loss_dict

    def pretrain(self):
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
            global_step = self.pretrain_one_epoch(global_step, epoch)
            # Evaluate
            if epoch > 0 and (epoch+1) % self.eval_interval == 0:
                pq, loss = self.evaluate_one_epoch()
                if pq > self.BEST_PQ:
                    self.BEST_PQ = pq
                    self.save_model(epoch, loss)
                self.PERFORMANCE_FOUT.write('epoch: ' + str(epoch) + '\n' + \
                                            'best: ' + str(self.BEST_PQ.item()) + '\n')
                self.PERFORMANCE_FOUT.flush()
            # Save model
            if (epoch+1) % self.save_interval == 0:
                self.save_model(epoch, loss)

    def pretrain_one_epoch(self, global_step, epoch):
        # self.adjust_learning_rate(EPOCH_CNT)
        self.bnm_scheduler.step()  # decay BN momentum
        self.segmentor.train()  # set model to training mode
        losses_dict = {
            "loss_ce": [],
            "loss_mask": [],
            "loss_dice": [],
            "semantic_loss_ce": [],
            "semantic_loss_mask": [],
            "semantic_loss_dice": [],
        }
        losses = []
        t_bar = tqdm(self.LABELED_DATALOADER)
        for batch_idx, batch_data_target in enumerate(t_bar):
            # 1.forward student model
            data, target, filename = batch_data_target
            self.optimizer.zero_grad()
            output = self.forward_segmentor(data, target, is_ema=False)
            # 2.compute loss and gradients, update parameters.
            loss, loss_dict = self.pretrain_calculate_loss(output, target)
            loss.backward()
            self.optimizer.step()
            global_step += 1
            # 3.log statistics
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

        self.lr_scheduler.step()
        return global_step
    
    def evaluate_one_epoch(self):
        stat_dict = {}  # collect statistics
        self.segmentor.eval()  # set model to eval mode (for bn and dp)
        losses = []
        top1, top2, top3 = [], [], []
        t_bar = tqdm(self.TEST_DATALOADER)
        for batch_idx, batch_data_target in enumerate(t_bar):
            # 1.forward frozen student model
            data, target, filename = batch_data_target
            raw_coordinates = data.features[:, -3:]
            inverse_maps = data.inverse_maps
            target_full = data.target_full
            original_colors = data.original_colors
            data_idx = data.idx
            original_normals = data.original_normals
            original_coordinates = data.original_coordinates
            original_labels = data.original_labels
            with torch.no_grad():
                output, raw_coordinates = self.forward_segmentor(data, target, is_ema=False, is_eval=True)
                loss, loss_dict = self.pretrain_calculate_loss(output, target)
                losses.append(loss.detach().cpu().item())

            pred_results, top1s, top2s, top3s = self.postprocess(output, target, raw_coordinates, filename, inverse_maps, original_labels)
            top1.extend(top1s)
            top2.extend(top2s)
            top3.extend(top3s)
            self.compute_PQ(pred_results, inverse_maps, original_labels, original_coordinates)
            # 2.convert masks to point-level, calculate scores, sort and filter 
            # all_pred_masks, all_heatmaps, all_pred_classes, all_pred_scores, all_semantic_masks, all_semantic_heatmaps = self.postprocess(
            #     output, target, target_full, inverse_maps, raw_coordinates
            #     )
            # self.calculate_PQ(
            #     all_pred_masks, 
            #     all_heatmaps, 
            #     all_pred_classes, 
            #     all_pred_scores, 
            #     all_semantic_masks, 
            #     all_semantic_heatmaps,
            #     original_labels)
            
        # Log statistics
        # for key in sorted(stat_dict.keys()):
        #     self.log_string('eval mean %s: %f' % (key, stat_dict[key] /
        #                                      (float(batch_idx + 1))))
        mean_loss = sum(losses)/len(losses)
        t_bar.set_postfix(mean_loss=f"{mean_loss:.4f}")
        PQ, SQ, RQ, all_pq, all_sq, all_rq = self.evaluator.getPQ()
        iou, all_iou = self.evaluator.getSemIoU()
        print("PQ:", PQ)
        print("SQ:", SQ)
        print("RQ:", RQ)
        print("IoU:", iou.item(), iou.item() == 0.5476190476190476)
        for i, (pq, sq, rq, iou) in enumerate(zip(all_pq, all_sq, all_rq, all_iou)):
            if i not in CLASS_DICT.keys():
                continue
            print("Class", CLASS_DICT[i].ljust(16), "\t",
                   "PQ:", "{:.4f}".format(pq.item()), 
                   "SQ:", "{:.4f}".format(sq.item()),
                    "RQ:", "{:.4f}".format(rq.item()),
                      "IoU:", "{:.4f}".format(iou.item()))
        print(top1)
        print(sum(top1)/len(top1))
        print(sum(top2)/len(top2))
        print(sum(top3)/len(top3))
        wandb.log({"eval_mean_loss:": mean_loss})
        wandb.log({"PQ": PQ})
        wandb.log({"mIoU": iou})
        return PQ, loss

    # def postprocess(self, output, target, target_full, inverse_maps, raw_coords):
        self.decoder_id = -1 # indicate the layer where we interest
        train_on_segments = self.segmentor.module.train_on_segments
        num_queries = self.segmentor.module.num_queries  # indicate the number of semantic queries
        num_classes = self.segmentor.module.num_classes + 1 # add class for uninterested
        all_pred_classes = list()
        all_pred_masks = list()
        all_pred_scores = list()
        all_heatmaps = list()
        all_semantic_masks = list()
        all_semantic_heatmaps = list()
        all_query_pos = list()
        prediction = output["aux_outputs"]
        prediction.append(
            {
                "pred_logits": output["pred_logits"],
                "pred_masks": output["pred_masks"],
            }
        )

        prediction[self.decoder_id]["pred_logits"] = torch.functional.F.softmax(
            prediction[self.decoder_id]["pred_logits"], dim=-1
        )[..., :-1]


        offset_coords_idx = 0
        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            
            # 1.convert segment-level masks to voxel-level masks, detach and push to cpu
            if train_on_segments:
                masks = (
                    prediction[self.decoder_id]["pred_masks"][bid]
                    .detach()
                    .cpu()[target[bid]["point2segment"].cpu()]
                )
            else:
                masks = (
                    prediction[self.decoder_id]["pred_masks"][bid]
                    .detach()
                    .cpu()
                )
            logits = (
                prediction[self.decoder_id]["pred_logits"][bid]
                .detach()
                .cpu()
            )
            num_masks = prediction[self.decoder_id]["pred_logits"][bid].shape[0]
            semantic_masks = masks[:,num_queries:]
            masks = masks[:,:num_queries]
            logits = logits[:num_queries,:]
            
            # 2.calculate scores and pick up topk masks 
            if self.test_config.use_dbscan:
                print("----------------------use DBSCAN to cluster or merge masks----------------------")
                new_preds = {
                    "pred_masks": list(),
                    "pred_logits": list(),
                }

                curr_coords_idx = masks.shape[0]
                curr_coords = raw_coords[
                            offset_coords_idx: curr_coords_idx + offset_coords_idx
                            ]
                offset_coords_idx += curr_coords_idx

                num_instance_masks = masks.shape[1]-num_classes
                print(f"number of pred instances:{num_instance_masks}")
                for curr_query in range(masks.shape[1]):
                    if curr_query >= num_instance_masks:
                        new_preds["pred_masks"].append(
                                    masks[:,curr_query]
                                )
                        new_preds["pred_logits"].append(
                                    prediction[self.decoder_id][
                                        "pred_logits"
                                    ][bid, curr_query]
                                )
                        continue
                    curr_masks = masks[:, curr_query] > 0
                    if curr_coords[curr_masks].shape[0] > 0:
                        clusters = (
                            DBSCAN(
                                eps=self.test_config.dbscan_eps,
                                min_samples=self.test_config.dbscan_min_points,
                                n_jobs=-1,
                            )
                            .fit(curr_coords[curr_masks])
                            .labels_
                        )

                        new_mask = torch.zeros(curr_masks.shape, dtype=int)
                        new_mask[curr_masks] = (
                                torch.from_numpy(clusters) + 1
                        )

                        for cluster_id in np.unique(clusters):
                            original_pred_masks = masks[:, curr_query]
                            if cluster_id != -1:
                                new_preds["pred_masks"].append(
                                    original_pred_masks
                                    * (new_mask == cluster_id + 1)
                                )
                                new_preds["pred_logits"].append(
                                    prediction[self.decoder_id][
                                        "pred_logits"
                                    ][bid, curr_query]
                                )

                scores, masks, classes, heatmap = self.get_mask_and_scores(
                    torch.stack(new_preds["pred_logits"]).cpu(),
                    torch.stack(new_preds["pred_masks"]).T,
                    self.device,
                    len(new_preds["pred_logits"]),
                    num_classes,
                )
            else:
                sorted_masks, sorted_heatmap, sort_classes, sort_scores_values = get_mask_and_scores(
                    logits,
                    masks,
                    self.device,
                    num_queries,
                    num_classes,
                )

            # 3.convert voxel-level masks to point-level masks
            semantic_heatmaps = semantic_masks.float().sigmoid()
            semantic_masks = (semantic_masks > 0).float()
            semantic_masks = get_full_res_mask(
                semantic_masks,
                inverse_maps[bid],
                target_full[bid]["point2segment"]
            )
            semantic_heatmaps = get_full_res_mask(
                semantic_heatmaps,
                inverse_maps[bid],
                target_full[bid]["point2segment"],
                is_heatmap=True,
            )
            sorted_masks = get_full_res_mask(
                sorted_masks,
                inverse_maps[bid],
                target_full[bid]["point2segment"],
            )
            sorted_heatmap = get_full_res_mask(
                sorted_heatmap,
                inverse_maps[bid],
                target_full[bid]["point2segment"],
                is_heatmap=True,
            )
            semantic_masks = semantic_masks.numpy()
            semantic_heatmaps = semantic_heatmaps.numpy()
            sorted_masks = sorted_masks.numpy()
            sorted_heatmap = sorted_heatmap.numpy()
            # if backbone_features is not None:
            #     backbone_features = self.get_full_res_mask(
            #         torch.from_numpy(backbone_features),
            #         inverse_maps[bid],
            #         target_full[bid]["point2segment"],
            #         is_heatmap=True,
            #     )
            #     print(backbone_features.shape)
            #     backbone_features = backbone_features.numpy()

            if self.test_config.nms:
                classes, masks, scores, heatmap = NMS(
                    sorted_masks, 
                    sorted_heatmap, 
                    sort_classes, 
                    sort_scores_values, 
                    scores_threshold=0., 
                    iou_threshold=0.
                )
                all_pred_classes.append(classes)
                all_pred_masks.append(masks)
                all_pred_scores.append(scores)
                all_heatmaps.append(heatmap)
            else:
                all_pred_classes.append(sort_classes)
                all_pred_masks.append(sorted_masks)
                all_pred_scores.append(sort_scores_values)
                all_heatmaps.append(sorted_heatmap)
            all_semantic_masks.append(semantic_masks)
            all_semantic_heatmaps.append(semantic_heatmaps)

        return all_pred_masks, all_heatmaps, all_pred_classes, all_pred_scores, all_semantic_masks, all_semantic_heatmaps

    # def calculate_PQ(self, all_pred_masks, all_heatmaps, all_pred_classes, all_pred_scores, all_semantic_masks, all_semantic_heatmaps, original_labels):
        for bid in range(len(all_pred_masks)):
            instance_masks = all_pred_masks[bid]
            instance_scores = all_pred_scores[bid].cpu().numpy()
            instance_labels = all_pred_classes[bid].cpu().numpy()
            semantic_masks = all_semantic_masks[bid]
            semantic_heatmaps = all_semantic_heatmaps[bid]
            semantic_map, instance_map = create_map(
                semantic_masks, semantic_heatmaps,
                instance_masks, instance_labels, instance_scores
                                                    )
            semantic_gt_map = original_labels[bid][:,0]
            semantic_gt_map[semantic_gt_map==255]=20  # convert 255 to 20 to calculate cm
            instance_gt_map = original_labels[bid][:,1]
            # need to input one scene as batch because of the different points numbers among different scenes
            # add a new axis as pseudo batch axis
            self.evaluator.addBatch(semantic_map[None,:], instance_map[None,:], 
                                    semantic_gt_map[None,:], instance_gt_map[None,:])
        
    def postprocess(self, output, target, raw_coords, scene_ids, inverse_maps, original_labels):
        """
        raw_coords: raw_voxel_coordinates
        """
        # class_weights = torch.tensor(self.test_config.class_weights, device=output["pred_logits"].device)
        # logits = logits * class_weights
        # output["pred_logits"] = output["pred_logits"]*class_weights
        top1s, top2s, top3s = [], [], []

        self.decoder_id = -1 # indicate the layer where we are interested
        train_on_segments = self.segmentor.module.train_on_segments
        prediction = output["aux_outputs"]
        prediction.append(
            {
                "pred_logits": output["pred_logits"],
                "pred_masks": output["pred_masks"],
            }
        )
        # prediction[self.decoder_id]["pred_logits"] = torch.functional.F.softmax(
        #     prediction[self.decoder_id]["pred_logits"], dim=-1)[..., :-1]  # convert logits to scores(with softmax), exclude uninterested class
        
        # 将 Softmax 替换为 Sigmoid
        prediction[self.decoder_id]["pred_logits"] = torch.functional.F.sigmoid(
            prediction[self.decoder_id]["pred_logits"])[..., :-1]
        
        prediction_results = []
        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            # 1.convert segment-level masks to voxel-level masks, detach and push to cpu
            if train_on_segments:
                masks = (
                    prediction[self.decoder_id]["pred_masks"][bid]
                    .detach()
                    .cpu()[target[bid]["point2segment"].cpu()]
                )
            else:
                masks = (
                    prediction[self.decoder_id]["pred_masks"][bid]
                    .detach()
                    .cpu()
                )
            logits = (
                prediction[self.decoder_id]["pred_logits"][bid]
                .detach()
                .cpu()
            )

            # statistics
            print(masks.shape)
            print(inverse_maps[bid].shape)
            binary_masks = masks.T[:,inverse_maps[bid]]
            binary_masks = (binary_masks.float().sigmoid()>0.5).numpy()[:80,:]
            semantic_gt_map = original_labels[bid][:,0]
            semantic_gt_map[semantic_gt_map==255]=20  # convert 255 to 20 to calculate cm
            print(np.unique(semantic_gt_map))
            print(semantic_gt_map.shape)
            out_masks = (semantic_gt_map+1)*binary_masks
            print(out_masks.shape)
            mask_sizes = binary_masks.sum(-1)+1
            print(mask_sizes)
            top1, top2, top3 = self.statistics(out_masks,mask_sizes)
            top1s.append(top1)
            top2s.append(top2)
            top3s.append(top3)

            # 2.use DBSCAN to postprocess predictions
            if self.test_config.use_dbscan:
                print("----------------------use DBSCAN to cluster or merge masks----------------------")
                num_classes = 20
       
                offset_coords_idx = 0
                new_preds = {
                    "pred_masks": list(),
                    "pred_logits": list(),
                }

                curr_coords_idx = masks.shape[0]
                curr_coords = raw_coords[offset_coords_idx: curr_coords_idx + offset_coords_idx]
                offset_coords_idx += curr_coords_idx
                num_instance_masks = masks.shape[1]-num_classes
                for curr_query in range(masks.shape[1]):
                    if curr_query >= num_instance_masks:
                        new_preds["pred_masks"].append(
                                    masks[:,curr_query]
                                )
                        new_preds["pred_logits"].append(
                                    prediction[self.decoder_id][
                                        "pred_logits"
                                    ][bid, curr_query]
                                )
                        continue
                    curr_masks = masks[:, curr_query] > 0
                    if curr_coords[curr_masks].shape[0] > 0:
                        clusters = (
                            DBSCAN(
                                eps=self.test_config.dbscan_eps,
                                min_samples=self.test_config.dbscan_min_points,
                                n_jobs=1,
                            )
                            .fit(curr_coords[curr_masks])
                            .labels_
                        )

                        new_mask = torch.zeros(curr_masks.shape, dtype=int)
                        new_mask[curr_masks] = (
                                torch.from_numpy(clusters) + 1
                        )

                        for cluster_id in np.unique(clusters):
                            original_pred_masks = masks[:, curr_query]
                            if cluster_id != -1:
                                new_preds["pred_masks"].append(
                                    original_pred_masks
                                    * (new_mask == cluster_id + 1)
                                )
                                new_preds["pred_logits"].append(
                                    prediction[self.decoder_id][
                                        "pred_logits"
                                    ][bid, curr_query]
                                )
                
                logits = torch.stack(new_preds["pred_logits"]).cpu()
                masks = torch.stack(new_preds["pred_masks"]).T
                print(f"----------------------after DBSCAN, There are {masks.shape[1]} masks ----------------------")

            prediction_results.append({"scene_id":scene_ids[bid], "masks": masks, "logits":logits})
            print(f"target labels:\n")
            print(target[bid]["labels"])
            print(torch.unique(target[bid]["labels"][:-20]))
        return prediction_results, top1s, top2s, top3s

    def statistics(self, numpy_array, mask_size):
        # 定义一个函数，用于计算一个数组中的元素频率并返回最高的三个元素及其频率
        def top_n_frequencies(arr, top_n=3):
            # 计算唯一值和其频率
            unique_values, counts = np.unique(arr, return_counts=True)
            
            # 将频率按降序排列并获取排序的索引
            sorted_indices = np.argsort(-counts)  # 负号表示降序排列
            
            # 获取频率最高的前 top_n 个值及其频率
            top_values = unique_values[sorted_indices][:top_n]
            top_counts = counts[sorted_indices][:top_n]
            
            # 如果 top_values 和 top_counts 的长度不足 top_n，用 -1 或 0 进行填充
            if len(top_values) < top_n:
                top_values = np.pad(top_values, (0, top_n - len(top_values)), 'constant', constant_values=-1)
                top_counts = np.pad(top_counts, (0, top_n - len(top_counts)), 'constant', constant_values=0)
            
            return top_values, top_counts

        # 使用 NumPy 的 apply_along_axis 函数沿行轴（axis=1）并行计算
        # 由于 top_n_frequencies 返回了两个数组，因此我们需要分别存储这两个结果
        top_n=4
        top_values_and_counts = np.apply_along_axis(lambda row: np.array(top_n_frequencies(row, top_n=top_n)).flatten(), axis=1, arr=numpy_array)
        
        # 将结果重新拆分为 top_values 和 top_counts
        top_values = top_values_and_counts[:, :top_n]
        top_counts = top_values_and_counts[:, top_n:]

        # 打印结果
        for i in range(numpy_array.shape[0]):
            print(f"\n第 {i + 1} 行的前 4 个频率最高的值及其频率:")
            for j in range(top_n):
                value = top_values[i, j]
                count = top_counts[i, j]
                print(f"值: {value} 频率: {count}")
        
        top_1 = np.mean(top_counts[:,1]/mask_size)
        top_2 = np.mean(top_counts[:,2]/mask_size)
        top_3 = np.mean(top_counts[:,3]/mask_size) 
        # print(np.mean(top_counts[:,1]/(top_counts[:,2]+1)))
        return top_1, top_2, top_3

    def compute_PQ(self, prediction_results, inverse_maps, original_labels, original_coordinates):
        # self.decoder_id = -1 # indicate the layer where we interest
        # train_on_segments = self.segmentor.module.train_on_segments
        # class_weights = torch.tensor(self.test_config.class_weights, device=output["pred_logits"].device)
        # # logits = logits * class_weights
        # # output["pred_logits"] = output["pred_logits"]*class_weights
        # prediction = output["aux_outputs"]
        # prediction.append(
        #     {
        #         "pred_logits": output["pred_logits"],
        #         "pred_masks": output["pred_masks"],
        #     }
        # )
        # # convert logits to scores(with softmax), exclude uninterested class
        # prediction[self.decoder_id]["pred_logits"] = torch.functional.F.softmax(
        #     prediction[self.decoder_id]["pred_logits"], dim=-1)[..., :-1]

        # for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
        #     # 1.convert segment-level masks to voxel-level masks, detach and push to cpu
        #     if train_on_segments:
        #         masks = (
        #             prediction[self.decoder_id]["pred_masks"][bid]
        #             .detach()
        #             .cpu()[target[bid]["point2segment"].cpu()]
        #         )
        #     else:
        #         masks = (
        #             prediction[self.decoder_id]["pred_masks"][bid]
        #             .detach()
        #             .cpu()
        #         )
        #     logits = (
        #         prediction[self.decoder_id]["pred_logits"][bid]
        #         .detach()
        #         .cpu()
        #     )
        #     print("target labels:\n")
        #     print(target[bid]["labels"])
        #     print(torch.unique(target[bid]["labels"][:-20]))
    

        #     # 2.use DBSCAN to postprocess predictions
        #     if self.test_config.use_dbscan:
        #         print("----------------------use DBSCAN to cluster or merge masks----------------------")
        #         num_classes = self.segmentor.module.num_classes
       
        #         offset_coords_idx = 0
        #         new_preds = {
        #             "pred_masks": list(),
        #             "pred_logits": list(),
        #         }

        #         curr_coords_idx = masks.shape[0]
        #         curr_coords = raw_coords[offset_coords_idx: curr_coords_idx + offset_coords_idx]
        #         offset_coords_idx += curr_coords_idx
        #         num_instance_masks = masks.shape[1]-num_classes
        #         print(f"number of pred instances:{num_instance_masks}")
        #         for curr_query in range(masks.shape[1]):
        #             if curr_query >= num_instance_masks:
        #                 new_preds["pred_masks"].append(
        #                             masks[:,curr_query]
        #                         )
        #                 new_preds["pred_logits"].append(
        #                             prediction[self.decoder_id][
        #                                 "pred_logits"
        #                             ][bid, curr_query]
        #                         )
        #                 continue
        #             curr_masks = masks[:, curr_query] > 0
        #             if curr_coords[curr_masks].shape[0] > 0:
        #                 clusters = (
        #                     DBSCAN(
        #                         eps=self.test_config.dbscan_eps,
        #                         min_samples=self.test_config.dbscan_min_points,
        #                         n_jobs=1,
        #                     )
        #                     .fit(curr_coords[curr_masks])
        #                     .labels_
        #                 )

        #                 new_mask = torch.zeros(curr_masks.shape, dtype=int)
        #                 new_mask[curr_masks] = (
        #                         torch.from_numpy(clusters) + 1
        #                 )

        #                 for cluster_id in np.unique(clusters):
        #                     original_pred_masks = masks[:, curr_query]
        #                     if cluster_id != -1:
        #                         new_preds["pred_masks"].append(
        #                             original_pred_masks
        #                             * (new_mask == cluster_id + 1)
        #                         )
        #                         new_preds["pred_logits"].append(
        #                             prediction[self.decoder_id][
        #                                 "pred_logits"
        #                             ][bid, curr_query]
        #                         )
                
        #         logits = torch.stack(new_preds["pred_logits"]).cpu()
        #         masks = torch.stack(new_preds["pred_masks"]).T
        for bid in range(len(prediction_results)):
            scene_id = prediction_results[bid]["scene_id"]
            masks = prediction_results[bid]["masks"]
            logits = prediction_results[bid]["logits"]
            # semantic_map_ = self.map_creator.pred_sem(masks[:, -20:].T, inverse_maps[bid])
            # print(semantic_map_.shape)
            # pred_semantic_masks = masks[:, -20:].T
            # pred_semantic_masks = pred_semantic_masks[:,inverse_maps[bid]]
            # pred_semantic_masks = pred_semantic_masks.sigmoid().numpy()
            
            # semantic_map, instance_map = self.map_creator.mask2former_pred_pan(masks.T, logits, inverse_maps[bid])
            semantic_map, instance_map = self.map_creator.pred_pan(masks.T, logits, inverse_maps[bid])

            semantic_map = semantic_map.cpu().numpy()
            instance_map = instance_map.cpu().numpy()
            # semantic_map[semantic_map==-1]=20
            # instance_map[semantic_map==-1]=101
            print(f"floor is occupied {semantic_map[semantic_map==1].shape[0]/semantic_map.shape[0]} area")
            print(f"wall is occupied {semantic_map[semantic_map==0].shape[0]/semantic_map.shape[0]} area")
            print(f"void is occupied {semantic_map[semantic_map==20].shape[0]/semantic_map.shape[0]} area")

            semantic_gt_map = original_labels[bid][:,0]
            semantic_gt_map[semantic_gt_map==255]=20  # convert 255 to 20 to calculate cm
            instance_gt_map = original_labels[bid][:,1]
            instance_gt_map[semantic_gt_map==0]=100  # merge the wall gt into a semantic gt
            instance_gt_map[semantic_gt_map==1]=101  # merge the floor gt into a semantic gt

            # need to input one scene as batch because of the different points numbers among different scenes
            # add a new axis as pseudo batch axis
            self.evaluator.addBatch(semantic_map[None,:], instance_map[None,:], 
                                    semantic_gt_map[None,:], instance_gt_map[None,:])
            # self.visualization_save(scene_id[bid], 
            #                         original_coordinates[bid], 
            #                         instance_map, 
            #                         instance_gt_map, 
            #                         pred_semantic_masks, 
            #                         semantic_gt_map)
            # exit()
            
            

    def monitor_memory(self):
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        available = reserved - allocated
        print(f"Allocated: {allocated / 1024**3:.2f} GB")
        print(f"Reserved: {reserved / 1024**3:.2f} GB")
        print(f"Available: {available / 1024**3:.2f} GB")

    def visualization_save(self, scene_id, coordinates, ins_pred, ins_gt, sem_pred, sem_gt):
        save_path = f"{self.LOG_DIR}/{scene_id}.npz"
        print(save_path)
        np.savez(save_path, coordinates=coordinates, ins_pred=ins_pred, ins_gt=ins_gt, sem_pred=sem_pred, sem_gt=sem_gt)
        print("successfuly saved.")

import torch.multiprocessing as mp
import torch.distributed as dist
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
    trainer = TrainerBase(args, segmentor=model, 
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
        trainer.pretrain()
    elif args.mode == "eval":
        trainer.evaluate_one_epoch()
    # trainer.train()
    wandb.finish()
    


if __name__ == '__main__':
    main()