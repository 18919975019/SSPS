from Mask3D.models.matcher import HungarianMatcher
from Mask3D.models.criterion import SetCriterion
import torch
from torch import nn

NUM_CLASSES = 18

#TODO: need to pass the config which specifies the select criterion
def topk_selection(config, pred_logits, pred_masks):
    """
    :param pred_logits: predicted logits for 1 scene, (n_queries x n_classes)
    :param pred_masks: predicted masks for 1 scene, (n_voxels x n_queries)
    :return:
    """
    num_queries = pred_logits.shape[0]
    num_classes = config.num_sem_cls  # count all semantic classes, including thing and stuff, excluding uninterested
    stuff_num_classes = len(config.stuff_cls)
    thing_num_classes = num_classes - stuff_num_classes
    labels = (
        torch.arange(stuff_num_classes, num_classes, device=pred_masks.device)
        .unsqueeze(0)
        .repeat(num_queries, 1)
        .flatten(0, 1)
    )
    cls_scores = pred_logits[:,stuff_num_classes:]
    assert labels.shape == cls_scores.flatten(0, 1).shape, "check the cls scores."
    topk_cls_scores, topk_indices = cls_scores.flatten(0, 1).topk(
            config.topk_per_scene, sorted=True
        )
    topk_labels = labels[topk_indices]
    topk_indices = topk_indices // thing_num_classes
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
    return sorted_masks.T, sorted_heatmap.T, sort_labels, sort_scores_values

def NMS(config, sorted_masks, sorted_heatmap, sort_classes, sort_scores_values, iou_threshold=0.5):
    keep_instances = set()
    pairwise_overlap = sorted_masks.T @ sorted_masks
    normalization = pairwise_overlap.max(axis=0)[0]
    norm_overlaps = pairwise_overlap / normalization.unsqueeze(0)
    for instance_id in range(norm_overlaps.shape[0]):
        # filter out unlikely masks and nearly empty masks
        # if not(sort_scores_values[instance_id] < 0.3 or sorted_masks[:, instance_id].sum() < 500):
        if not (sort_scores_values[instance_id] < config.ins_score_thr):
            # check if mask != empty
            if not sorted_masks[:, instance_id].sum() == 0.0:
                overlap_ids = set(
                    torch.nonzero(norm_overlaps[instance_id, :] > iou_threshold).squeeze(-1)
                )
                if len(overlap_ids) == 0:
                    keep_instances.add(instance_id)
                else:
                    if instance_id == min(overlap_ids):
                        keep_instances.add(instance_id)

    keep_instances = sorted(list(keep_instances))
    return sorted_masks[:, keep_instances].T, sorted_heatmap[:, keep_instances].T,\
            sort_classes[keep_instances], sort_scores_values[keep_instances]

def get_mask3d_segmentation_loss(output, target, train_on_segments):
    if train_on_segments:
        mask_type = "segment_mask"
    else:
        mask_type = "masks"
    matcher = HungarianMatcher(cost_class=2.,cost_mask=5.,cost_dice=2.,num_points=-1)
    weight_dict = {
        "loss_ce": matcher.cost_class,
        "loss_mask": matcher.cost_mask,
        "loss_dice": matcher.cost_dice,
    }

    ignore_mask_idx = []
    aux_weight_dict = {}
    for i in range(12):
        if i not in ignore_mask_idx:
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in weight_dict.items()}
            )
        else:
            aux_weight_dict.update(
                {k + f"_{i}": 0.0 for k, v in weight_dict.items()}
            )
    weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(num_classes=21, 
                             matcher=matcher,
                             weight_dict=weight_dict,
                             eos_coef=0.1,
                             losses=['labels','masks'],
                             num_points=matcher.num_points,
                             oversample_ratio=3.0,
                             importance_sample_ratio=0.75,
                             class_weights=-1
                             )
    losses = criterion(output, target, mask_type=mask_type)
    for k in list(losses.keys()):
        if k in criterion.weight_dict:
            losses[k] *= criterion.weight_dict[k]
        else:
            # remove this loss if not specified in `weight_dict`
            losses.pop(k)
    return sum(losses.values()), losses

from loss.panoptic_criterion import PanopticCriterion
def get_segmentation_loss(output, target, train_on_segments):
    # Define which loss will be kept in the backward pass
    weight_dict = {
        "loss_ce": 0.5,
        "loss_mask": 1.0,
        "loss_dice": 1.0,
        "semantic_loss_mask": 5.,
        # "semantic_loss_ce": matcher.cost_class,
        # "semantic_loss_dice": 5.,
    }
    
    # 1.create matcher
    matcher = HungarianMatcher(cost_class=weight_dict["loss_ce"],
                               cost_mask=weight_dict["loss_mask"],
                               cost_dice=weight_dict["loss_dice"],
                               num_points=-1, ignore_class=20)
    # 2.create loss weight dict
    ignore_mask_idx = []
    aux_weight_dict = {}
    for i in range(12):
        if i not in ignore_mask_idx:
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in weight_dict.items()}
            )
        else:
            aux_weight_dict.update(
                {k + f"_{i}": 0.0 for k, v in weight_dict.items()}
            )
    weight_dict.update(aux_weight_dict)
    
    # 3.calculate loss
    criterion = PanopticCriterion(num_classes=NUM_CLASSES, 
                                    matcher=matcher,
                                    weight_dict=weight_dict,
                                    eos_coef=0.1,
                                    instance_losses=['labels','masks'],
                                    semantic_losses=['masks'],
                                    num_points=matcher.num_points,
                                    oversample_ratio=3.0,
                                    importance_sample_ratio=0.75,
                                    class_weights=-1
                                    )
    
    # map the instance cls(2-19) to (0-17)
    for t in target:
        t['labels'][:-20] -= 2
    
    if train_on_segments:
        mask_type = "segment_mask"
    else:
        mask_type = "masks"
    losses = criterion(output, target, mask_type=mask_type)
    # 4.weight and sum losses
    for k in list(losses.keys()):
        if k in criterion.weight_dict:
            losses[k] *= criterion.weight_dict[k]
        else:
            # remove this loss if not specified in `weight_dict`
            losses.pop(k)
    return sum(losses.values()), losses

def get_thing_stuff_loss(output, target, train_on_segments):
    # 1.create matcher
    weight_dict = {
        "loss_ce": 0.5,
        "loss_mask": 1.0,
        "loss_dice": 1.0,
        "semantic_loss_mask": 5.,
        # "semantic_loss_ce": matcher.cost_class,
        # "semantic_loss_dice": 5.,
    }
    matcher = HungarianMatcher(cost_class=weight_dict["loss_ce"],
                               cost_mask=weight_dict["loss_mask"],
                               cost_dice=weight_dict["loss_dice"],
                               num_points=-1, ignore_class=20)
    # 2.create loss weight dict
    ignore_mask_idx = []
    aux_weight_dict = {}
    for i in range(12):
        if i not in ignore_mask_idx:
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in weight_dict.items()}
            )
        else:
            aux_weight_dict.update(
                {k + f"_{i}": 0.0 for k, v in weight_dict.items()}
            )
    weight_dict.update(aux_weight_dict)
    
    # 3.calculate loss
    from loss.panoptic_criterion import PanopticCriterionThingStuff
    criterion = PanopticCriterionThingStuff(num_classes=NUM_CLASSES, 
                                    matcher=matcher,
                                    weight_dict=weight_dict,
                                    eos_coef=0.1,
                                    losses=['labels','masks'],
                                    num_points=matcher.num_points,
                                    oversample_ratio=3.0,
                                    importance_sample_ratio=0.75,
                                    class_weights=-1
                                    )
    if train_on_segments:
        mask_type = "segment_mask"
    else:
        mask_type = "masks"
    losses = criterion(output, target, mask_type=mask_type)
    # 4.weight and sum losses
    for k in list(losses.keys()):
            if k in criterion.weight_dict:
                losses[k] *= criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
    return sum(losses.values()), losses
    
def prepare_target(pseudo_labels, unique_maps, target, original_labels):
    import numpy as np
    """
    pseudo_labels: list of np, each represent a sample, with shape [n_points x 2], sem_label & ins_label
    pseudo_maps: list of np array, each represent a sample, with shape[n_points]
    """
    labels = []
    for i in range(len(pseudo_labels)):
        sem_ins_label = pseudo_labels[i][unique_maps[i]]
        seg_label = torch.from_numpy(original_labels[i][unique_maps[i]][:,-1]).unsqueeze(-1)
        label = torch.cat([sem_ins_label, seg_label], dim=-1)
        labels.append(label)
    input_dict = {"labels": labels}
    input_dict["segment2label"] = []
    for i in range(len(input_dict["labels"])):
        # TODO BIGGER CHANGE CHECK!!!
        _, ret_index, ret_inv = np.unique(input_dict["labels"][i][:, -1], return_index=True,
                                          return_inverse=True)
        input_dict["labels"][i][:, -1] = torch.from_numpy(ret_inv)
        # [4, 4, 0, 0, 0, 1, 2, 3] length = num_points_in_a_pcd
        input_dict["segment2label"].append(input_dict["labels"][i][ret_index][:, :-1])
        # [4, 0, 1, 2, 3] length = num_segments_in_a_pcd

    list_labels = input_dict["labels"]  # list:[bs x torch.tensor([num_voxels_per_pcd x 2])]
    
    from datasets.collate_function import get_masks
    pseudo_target = get_masks(list_labels,
                                list_segments=input_dict["segment2label"],
                                ignore_class_threshold=100,
                                filter_out_classes=[],
                                )
    for i in range(len(pseudo_target)):
        pseudo_target[i]["point2segment"] = input_dict["labels"][i][:, 2]
    return pseudo_target

def get_pseudo_target(ema_output, ema_target, train_on_segments, inverse_maps, unique_maps, original_labels, map_creator, raw_coords=None):
    pseudo_labels = [] 
    decoder_id = -1 # indicate the layer where we interest
    prediction = []
    prediction.append(
        {
            "pred_logits": ema_output["pred_logits"],
            "pred_masks": ema_output["pred_masks"],
        }
    )
    # convert logits to scores(with softmax), exclude uninterested class
    prediction[decoder_id]["pred_logits"] = torch.functional.F.softmax(
        prediction[decoder_id]["pred_logits"], dim=-1)[..., :-1]
    
    #TODO: need to add point2segment to map segments to voxels, is it possible to add target for ema_output?
    for bid in range(len(prediction[decoder_id]["pred_masks"])):
        # 1.convert segment-level masks to voxel-level masks, detach and push to cpu
        if train_on_segments:
            masks = (
                prediction[decoder_id]["pred_masks"][bid]
                .detach()
                .cpu()[ema_target[bid]["ema_point2segment"].cpu()]
            )
        else:
            masks = (
                prediction[decoder_id]["pred_masks"][bid]
                .detach()
                .cpu()
            )
        logits = (
            prediction[decoder_id]["pred_logits"][bid]
            .detach()
            .cpu()
        )
        # from SSPS.evaluator.MapCreator import MapCreator
        # from omegaconf import OmegaConf
        # map_config_dict = {
        #     "stuff_cls": [0, 1],
        #     "thing_cls": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        #     "num_sem_cls": 20,
        #     "topk_per_scene": 100,
        #     "ins_score_thr": 0.,
        #     "n_point_thr": 100,
        #     "obj_normalization": False,
        #     "obj_normalization_thr": 0.01,
        #     "nms": True
        # }
        # map_config = OmegaConf.create(map_config_dict)
        # map_creator = MapCreator(map_config, ema_output["pred_logits"].device)

        semantic_map, instance_map = map_creator.pred_pan(masks.T, logits, inverse_maps[bid])
        pseudo_label = torch.stack([semantic_map, instance_map], dim=-1)   
        pseudo_labels.append(pseudo_label)
        # masks.T, 
        # pred_masks = pred_masks.T[:-config.num_sem_cls, :],
        # pred_logits = pred_logits[:-config.num_sem_cls, :],
        # masks, heatmaps, labels, scores = topk_selection(pred_logits, pred_masks.T)
        # masks, heatmaps, labels, scores = NMS(masks.T, heatmaps.T, labels, scores)
        #TODO: need to make psedo-target masks with current voxel-level masks
    
    
    # for idx in range(num_labeled, len(ema_output["pred_logits"])):
    #     pred_masks = ema_output["pred_masks"][idx]
    #     target = {}
    #     pred_sem_cls = ema_output["pred_logits"][idx]
    #     pred_sem_cls = nn.Softmax(dim=-1)(pred_sem_cls)
    #     max_cls, argmax_cls = torch.max(pred_sem_cls, dim=-1)
    #     cls_mask = max_cls > filter_config['cls_threshold']
       
    #     target["labels"] = argmax_cls[cls_mask]
    #     print(target["labels"])
    #     target["segment_mask"] = pred_masks[:,cls_mask]
    #     print(target["segment_mask"].shape)
    #     target["masks"] = pred_masks[:, cls_mask]
    #     pseudo_targets.append(target)
    pseudo_target = prepare_target(pseudo_labels, unique_maps, ema_target, original_labels)
    return pseudo_target

def get_labeled_loss(output, target, num_labeled, train_on_segments):
    labeled_output = {
        "pred_masks":output["pred_masks"][:num_labeled],
        "pred_logits":output["pred_logits"][:num_labeled]      
                }
    labeled_target = target[:num_labeled]
    labeled_loss, labeled_loss_dict = get_segmentation_loss(labeled_output, labeled_target, train_on_segments)
    return labeled_loss, labeled_loss_dict

def get_unlabeled_loss(output, ema_output, num_labeled, train_on_segments, data, target, map_creator):
    unlabeled_output = {
        "pred_masks":output["pred_masks"][num_labeled:],
        "pred_logits":output["pred_logits"][num_labeled:]      
                }
    unlabeled_target = target[num_labeled:]
    unlabeled_ema_output = {
        "pred_masks":ema_output["pred_masks"][num_labeled:],
        "pred_logits":ema_output["pred_logits"][num_labeled:]
    }
    ema_inverse_maps = data.ema_inverse_maps[num_labeled:]
    unique_maps = data.unique_maps[num_labeled:]
    original_labels = data.original_labels[num_labeled:]
    # raw_coordinates = data.features[:, -3:]
    unlabeled_target = get_pseudo_target(unlabeled_ema_output, 
                                         unlabeled_target, 
                                         train_on_segments, 
                                         ema_inverse_maps, 
                                         unique_maps,
                                         original_labels,
                                         map_creator)
    unlabeled_loss, unlabeled_loss_dict = get_segmentation_loss(unlabeled_output, 
                                           unlabeled_target, 
                                           train_on_segments)
    return unlabeled_loss, unlabeled_loss_dict

