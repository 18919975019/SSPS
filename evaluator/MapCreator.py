import torch
import torch.nn.functional as F

class MapCreator:
    def __init__(self, test_config, device):
        self.device = device
        self.stuff_cls = test_config.stuff_cls
        self.thing_cls = test_config.thing_cls
        self.num_sem_cls = test_config.num_sem_cls
        self.topk_per_scene = test_config.topk_per_scene
        self.ins_score_thr = test_config.ins_score_thr
        self.n_point_thr = test_config.n_point_thr
        self.obj_normalization = test_config.obj_normalization
        self.obj_normalization_thr = test_config.obj_normalization_thr
        self.nms = test_config.nms
        self.nms_iou_threshold = test_config.nms_iou_threshold

    def create_semantic_map(self, pred_masks, inverse_map):
        semantic_map = self.pred_sem(pred_masks[-self.num_sem_cls:, :], inverse_map)
        return semantic_map

    def create_instance_map(self, pred_masks, pred_logits, inverse_map):
        instance_map = self.pred_inst(pred_masks[:-self.num_sem_cls, :],
                                  pred_logits[:-self.num_sem_cls, :],
                                  inverse_map)
        return instance_map

    def create_panoptic_map(self, pred_masks, pred_logits, inverse_map):
        semantic_map, instance_map = self.pred_pan(pred_masks, pred_logits, inverse_map)
        return semantic_map, instance_map

    def pred_inst(self, pred_masks, pred_logits, inverse_map):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: masks of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        # scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        # scores *= pred_scores
        #
        # labels = torch.arange(self.num_classes, device=scores.device).unsqueeze(0)\
        #     .repeat(self.decoder.num_queries - self.test_cfg.num_sem_cls, 1).flatten(0, 1)
        #
        # scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        # labels = labels[topk_idx]
        #
        # topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        # mask_pred = pred_masks
        # mask_pred = mask_pred[topk_idx]
        # torch.set_printoptions(
        #             precision=2,       # 小数点后四位
        #             threshold=float('inf'),  # 打印完整张量
        #             linewidth=100,     # 每行打印的最大字符数，增加这个值避免折行
        #             sci_mode=False)     # 关闭科学记数法
        print(f"---------------------- start postprocess... ----------------------")
        print(f"{pred_masks.shape[0]} masks are input:\n")

        masks, heatmaps, labels, scores = self.topk_selection(pred_logits, pred_masks.T)
        print(f"---------------------- after topk selection ----------------------")
        print(f"{masks.shape[0]} masks are reserved:\n")
        print(f"the labels:\n{labels}")
        print(f"the scores:\n{scores}")
        mask_pred_sigmoid = heatmaps
        # normalize scores with the pred object size
        if self.obj_normalization:
            mask_pred_thr = heatmaps > self.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        # use ins Score Threshold to filter out masks, then nms
        if self.nms:
            # kernel = self.test_cfg.matrix_nms_kernel
            # scores, labels, heatmaps, _ = mask_matrix_nms(
            #     heatmaps, labels, scores, kernel=kernel)
            masks, heatmaps, labels, scores = self.NMS(masks.T, heatmaps.T, labels, scores)
        print(f"---------------------- after NMS ----------------------")
        print(f"{masks.shape[0]} masks are reserved:\n")
        
        print(f"the labels:\n{labels}")
        print(f"the scores:\n{scores}")
        # mask_pred = heatmaps > self.test_cfg.sp_score_thr
        masks = masks[:, inverse_map]
        # use ins Num Points Threshold to filter out full res masks
        mask_points_num = masks.sum(1)
        n_point_mask = mask_points_num > self.n_point_thr
        scores = scores[n_point_mask]
        labels = labels[n_point_mask]
        masks = masks[n_point_mask]
        print(f"---------------------- after small masks filter ----------------------")
        print(f"{masks.shape[0]} masks are reserved:\n")
        print(f"the labels:\n{labels}")
        print(f"the scores:\n{scores}")
        
        print(f"---------------------- finish postprocess... ----------------------")
        return masks, labels, scores

    def pred_sem(self, pred_sem_masks, inverse_map):
        """Predict semantic masks for a single scene.

        Args:
            pred_sem_masks (Tensor): of shape (n_sem_classes, n_voxels).
            inverse_map (Tensor): of shape (n_raw_points,).
        Returns:
            Tensor: semantic preds of shape (n_raw_points,), each point is assigned an int indicating class
        """
        
        mask_pred = pred_sem_masks.sigmoid()
        mask_pred = mask_pred[:, inverse_map]
        seg_map = mask_pred.argmax(0)
        return seg_map

    def pred_pan(self, pred_masks, pred_logits, inverse_map):
        """Predict panoptic masks for a single scene.
        Args:
            pred_masks (Tensor): of shape (n_queries, n_voxels).
            pred_logits (Tensor): of shape (n_queries, n_classes + 1).
            inverse_maps (Tensor): of shape (n_raw_points,).
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        # 1. Get stuff semantic map
        stuff_cls = pred_masks.new_tensor(self.stuff_cls).long()
        # following sem_map is fully-filled with stuff classes, ex.(only contains 0 or 1)
        sem_map = self.pred_sem(pred_masks[-self.num_sem_cls + stuff_cls, :], inverse_map)  # full-stuff semantic mask
        sem_map_src_mapping = stuff_cls[sem_map]  # remap 0-start cls idx back to idx in stuff_cls
        # 2. Get thing semantic map and thing instance map
        # choose instance masks
        mask_pred, labels, scores = self.pred_inst(
            pred_masks[:-self.num_sem_cls, :],
            pred_logits[:-self.num_sem_cls, :],
            inverse_map)
        # filter out stuff masks
        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.thing_cls:
            cls_mask = labels == thing_cls
            thing_idxs = torch.logical_or(thing_idxs, cls_mask)
        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]
        # if this scene does not include any instances, directly return semantic segmentation map
        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map
        # sort instance predictions with ascending order, larger instance id represent more confident prediction
        scores, idxs = scores.sort()
        labels = labels[idxs]
    
        mask_pred = mask_pred[idxs]
        inst_idxs = torch.arange(0, mask_pred.shape[0], device=mask_pred.device).view(-1, 1)
        # insts of shape (n_instance, n_points), assign each instance mask an instance id
        insts = inst_idxs * mask_pred  # assign each instance mask a instance id
        # things_inst_mask, things_sem_mask of shape (n_instance,), 0 represent void area or stuff area
        things_inst_mask, idxs = insts.max(axis=0)  # assign each point ins id according to most confident mask
    
        things_sem_mask = labels[idxs]  # assign each point sem id according to most confident mask

    
        # filter out too small instance
        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.n_point_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0
        things_inst_mask = torch.unique(
            things_inst_mask, return_inverse=True)[1]
        # increase instance by number of stuff classes, distinguish them with stuff classes id
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        # void and stuff area was assigned the class of first instance, now correct it to 0
        things_sem_mask[things_inst_mask == 0] = 0
        # void and thing area was assigned the class of stuff, now correct it to 0, fill with thing classes
        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map_src_mapping += things_sem_mask
        # void and thing area was assigned the instance id of stuff, now correct it to 0, fill with thing instance id
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        semantic_map = sem_map_src_mapping
        instance_map = sem_map
        return semantic_map, instance_map

    def topk_selection(self, pred_logits, pred_masks):
        """
        :param pred_logits: predicted logits for 1 scene, (n_queries x n_classes)
        :param pred_masks: predicted masks for 1 scene, (n_voxels x n_queries)
        :return:
        """

        num_queries = pred_logits.shape[0]
        print(f"{num_queries} pred masks are used to select topk... ")
        num_classes = self.num_sem_cls  # count all semantic classes, including thing and stuff, excluding uninterested
        stuff_num_classes = len(self.stuff_cls)
        thing_num_classes = num_classes - stuff_num_classes
        
        labels = (
            torch.arange(thing_num_classes, device=pred_masks.device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )
        cls_scores = pred_logits
       
        assert labels.shape == cls_scores.flatten(0, 1).shape, "check the cls scores."

        topk_cls_scores, topk_indices = cls_scores.flatten(0, 1).topk(
                min(cls_scores.flatten(0, 1).shape[0], self.topk_per_scene), sorted=True
            )
        topk_labels = labels[topk_indices] + stuff_num_classes

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
        print(f"{sort_labels.shape[0]} pred masks are selected... ")
        print(f"the predicted labels are:\n")
        print(torch.unique(sort_labels))
        
        return sorted_masks.T, sorted_heatmap.T, sort_labels, sort_scores_values

    def NMS(self, sorted_masks, sorted_heatmap, sort_classes, sort_scores_values):
        keep_instances = set()
        pairwise_overlap = sorted_masks.T @ sorted_masks
        normalization = pairwise_overlap.max(axis=0)[0]
        norm_overlaps = pairwise_overlap / normalization.unsqueeze(0)

        for instance_id in range(norm_overlaps.shape[0]):
            # filter out unlikely masks and nearly empty masks
            # if not(sort_scores_values[instance_id] < 0.3 or sorted_masks[:, instance_id].sum() < 500):
            if not (sort_scores_values[instance_id] < self.ins_score_thr):
                # check if mask != empty
                if not sorted_masks[:, instance_id].sum() == 0.0:
                    overlap_ids = set(
                        torch.nonzero(norm_overlaps[instance_id, :] > self.nms_iou_threshold).squeeze(-1)
                    )
                    if len(overlap_ids) == 0:
                        keep_instances.add(instance_id)
                    else:
                        if instance_id == min(overlap_ids):
                            keep_instances.add(instance_id)

        keep_instances = sorted(list(keep_instances))
        return sorted_masks[:, keep_instances].T, sorted_heatmap[:, keep_instances].T,\
               sort_classes[keep_instances], sort_scores_values[keep_instances]
    
    def mask2former_pred_pan(self, pred_masks, pred_logits, inverse_map):
        """Predict panoptic masks for a single scene.
        Args:
            pred_masks (Tensor): of shape (n_queries, n_voxels).
            pred_logits (Tensor): of shape (n_queries, n_classes + 1).
            inverse_maps (Tensor): of shape (n_raw_points,).
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        # Get thing semantic map and thing instance map
        # choose instance masks
        full_mask_pred, full_labels, full_scores = self.pred_inst(
            pred_masks,
            pred_logits,
            inverse_map)
 
        # filter out stuff masks
        thing_idxs = torch.zeros_like(full_labels)
        for thing_cls in self.thing_cls:
            cls_mask = full_labels == thing_cls
            thing_idxs = torch.logical_or(thing_idxs, cls_mask)
 
        mask_pred = full_mask_pred[thing_idxs]
        scores = full_scores[thing_idxs]
        labels = full_labels[thing_idxs]
        
        stuff_mask_pred = full_mask_pred[~thing_idxs]
        stuff_scores = full_scores[~thing_idxs]
        stuff_labels = full_labels[~thing_idxs]
        stuff_scores, stuff_idxs = stuff_scores.sort()
        stuff_labels = stuff_labels[stuff_idxs]
        stuff_mask_pred = stuff_mask_pred[stuff_idxs]

        print(stuff_mask_pred.shape)
        print(stuff_scores)
        print(stuff_labels)
        if stuff_mask_pred.shape[0] != 0:
            stuff_score_map = stuff_mask_pred*stuff_scores.unsqueeze(-1)
            background_score_map = torch.zeros_like(stuff_mask_pred[0]).unsqueeze(0)
            print(stuff_score_map.shape)
            print(background_score_map.shape)
            print(torch.unique(stuff_score_map))
            print(torch.unique(background_score_map))
            score_map = torch.cat([stuff_score_map.T, background_score_map.T], dim=-1)
            stuff_labels = torch.cat((torch.tensor([-1]), stuff_labels))
            print(score_map.shape)
            print(stuff_labels)

            stuff_cls = pred_masks.new_tensor(self.stuff_cls).long()
            sem_map = stuff_labels[torch.argmax(score_map, dim=-1)]
            sem_map_src_mapping = sem_map.clone()
        else:
            stuff_cls = pred_masks.new_tensor(self.stuff_cls).long()
            sem_map = torch.full(mask_pred[0].shape, -1)
            sem_map_src_mapping = sem_map.clone()
        

        # if this scene does not include any instances, directly return semantic segmentation map
        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map
        # sort instance predictions with ascending order, larger instance id represent more confident prediction
        scores, idxs = scores.sort()
        labels = labels[idxs]
    
        mask_pred = mask_pred[idxs]
        inst_idxs = torch.arange(0, mask_pred.shape[0], device=mask_pred.device).view(-1, 1)
        # insts of shape (n_instance, n_points), assign each instance mask an instance id
        insts = inst_idxs * mask_pred  # assign each instance mask a instance id
        # things_inst_mask, things_sem_mask of shape (n_instance,), 0 represent void area or stuff area
        things_inst_mask, idxs = insts.max(axis=0)  # assign each point ins id according to most confident mask
    
        things_sem_mask = labels[idxs]  # assign each point sem id according to most confident mask

    
        # filter out too small instance
        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.n_point_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0
        things_inst_mask = torch.unique(
            things_inst_mask, return_inverse=True)[1]
        # increase instance by number of stuff classes, distinguish them with stuff classes id
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        # void and stuff area was assigned the class of first instance, now correct it to 0
        things_sem_mask[things_inst_mask == 0] = 0
        # void and thing area was assigned the class of stuff, now correct it to 0, fill with thing classes
        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map_src_mapping += things_sem_mask
        # void and thing area was assigned the instance id of stuff, now correct it to 0, fill with thing instance id
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        semantic_map = sem_map_src_mapping
        instance_map = sem_map
        return semantic_map, instance_map