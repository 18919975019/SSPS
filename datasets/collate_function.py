import MinkowskiEngine as ME
import numpy as np
import torch
NUM_CLASSES = 20

class VoxelizeCollate:
    def __init__(
            self,
            ignore_label=255,
            voxel_size=0.02,
            mode="test",
            small_crops=False,
            very_small_crops=False,
            batch_instance=False,
            probing=False,
            task="panoptic_segmentation",
            ignore_class_threshold=100,
            filter_out_classes=[],
            label_offset=0,
    ):
        self.task = task
        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset
        self.voxel_size = voxel_size
        self.ignore_label = ignore_label
        self.mode = mode
        self.batch_instance = batch_instance
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops
        self.probing = probing
        self.ignore_class_threshold = ignore_class_threshold

    def __call__(self, batch):
        """

        :param batch: [
                       [coordinates,
                        features,
                        original_labels,
                        inverse_maps,
                        full_res_coords,target_full,
                        original_colors,
                        original_normals,
                        original_coordinates,
                        idx],
                        target,
                        list of scan_id. ex.[“scene0636_00”,…,“scene0634_03”]
                      ]
        :return: batch: [NoGpu(
                        coordinates,
                        features,
                        original_labels,
                        inverse_maps,
                        full_res_coords,target_full,
                        original_colors,
                        original_normals,
                        original_coordinates,
                        idx),
                        target,
                        list of scan_id. ex.[“scene0636_00”,…,“scene0634_03”]
                      ]
        """
        return voxelize(batch, self.ignore_label, self.voxel_size, self.probing, self.mode,
                        task=self.task, ignore_class_threshold=self.ignore_class_threshold,
                        filter_out_classes=self.filter_out_classes, label_offset=self.label_offset)


def voxelize(batch, ignore_label, voxel_size, probing, mode, task,
             ignore_class_threshold, filter_out_classes, label_offset):
    # 1.create a batched data list, each item is as shape of (B x num_points)
    (coordinates,
     features,
     labels,
     inverse_maps,
     original_labels,
     original_colors,
     original_normals,
     original_coordinates,
     idx,
     full_res_coords,
     ema_coordinates,
     ema_features,
     ema_labels,
     ema_inverse_maps,
     unique_maps) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [])

    voxelization_dict = {
        "ignore_label": ignore_label,
        "return_index": True,
        "return_inverse": True,
    }
    ema_voxelization_dict = {
        "ignore_label": ignore_label,
        "return_index": True,
        "return_inverse": True,
    }
    for sample in batch:
        original_labels.append(sample[2])
        original_colors.append(sample[4])
        original_normals.append(sample[5])
        original_coordinates.append(sample[6])
        idx.append(sample[7])
        full_res_coords.append(sample[0])
        # Get voxel coordinates and voxel features
        coords = np.floor(sample[0] / voxel_size)  # voxelized coordinates
        voxelization_dict.update({"coordinates": torch.from_numpy(coords).to("cpu").contiguous(),
                                  "features": sample[1]})
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(**voxelization_dict)
        inverse_maps.append(inverse_map)
        sample_coordinates = coords[unique_map]
        sample_features = sample[1][unique_map]
        sample_labels = sample[2][unique_map]
        coordinates.append(torch.from_numpy(sample_coordinates).int())
        features.append(torch.from_numpy(sample_features).float())
        labels.append(torch.from_numpy(sample_labels).long())
        # print(coords)
        # print(augmentation(coords))
        # arr = augmentation(coords)
        # print(coords.shape)
        # print(arr.shape)
        # is_integer = (arr == arr.astype(int))
        # print(sum(is_integer))
        # print("end")
        # exit()
        # print(unique_map.shape)
        # print(sample_coordinates.shape)
        # print(sample_labels.shape)
        
        unique_maps.append(unique_map)
        
        # take raw coordinates and centralize&normalize
        raw_coords = sample[6]
        raw_coords -= sample[6].mean(0) 
        raw_coords += np.random.uniform(sample[6].min(0), sample[6].max(0)) / 2
        ema_coords = np.floor(raw_coords / voxel_size)  # voxelized coordinates
        
        ema_voxelization_dict.update({"coordinates": torch.from_numpy(ema_coords).to("cpu").contiguous(),
                                  "features": np.hstack((sample[4], raw_coords))})
        _, _, ema_unique_map, ema_inverse_map = ME.utils.sparse_quantize(**ema_voxelization_dict)
        ema_inverse_maps.append(ema_inverse_map)
        ema_sample_coordinates = ema_coords[ema_unique_map]
        ema_sample_features = np.hstack((sample[4], raw_coords))[ema_unique_map]
        ema_sample_labels = sample[2][ema_unique_map]
        ema_coordinates.append(torch.from_numpy(ema_sample_coordinates).float())
        ema_features.append(torch.from_numpy(ema_sample_features).float())
        ema_labels.append(torch.from_numpy(ema_sample_labels).long())
        # print("ema results")
        # print(ema_unique_map.shape)
        # print(ema_sample_coordinates.shape)
        # print(ema_sample_labels.shape)
        # print(f"voxel coords shape: {coords.shape}")
        # # 找到每个维度的最小值和最大值
        # min_values = coords.min(axis=0)
        # max_values = coords.max(axis=0)
        # # 输出结果
        # print(f"voxel空间的边界:")
        # print(f"x 轴范围: {min_values[0]} 到 {max_values[0]}")
        # print(f"y 轴范围: {min_values[1]} 到 {max_values[1]}")
        # print(f"z 轴范围: {min_values[2]} 到 {max_values[2]}")
        # print(f"ema_voxel coords shape: {coords.shape}")
        # min_values = ema_coords.min(axis=0)
        # max_values = ema_coords.max(axis=0)
        # # 输出结果
        # print(f"ema_voxel空间的边界:")
        # print(f"x 轴范围: {min_values[0]} 到 {max_values[0]}")
        # print(f"y 轴范围: {min_values[1]} 到 {max_values[1]}")
        # print(f"z 轴范围: {min_values[2]} 到 {max_values[2]}")
        # exit()

    # print("start")
    # print(coordinates[0].shape)
    # print(features[0].shape)
    # print(labels[0].shape)
    # print(len(coordinates))
    # print(len(features))
    # print(len(labels))

    # print(ema_coordinates[0].shape)
    # print(ema_features[0].shape)
    # print(ema_labels[0].shape)
    # print(len(ema_coordinates))
    # print(len(ema_features))
    # print(len(ema_labels))
    # print("finish")
        
    # 2.convert coordinates, features, labels to SparsTensor
    input_dict = {"coords": coordinates, "feats": features, "labels": labels}
    coordinates, features, labels = ME.utils.sparse_collate(**input_dict)
    ema_input_dict = {"coords": ema_coordinates, "feats": ema_features, "labels": ema_labels}
    ema_coordinates, ema_features, ema_labels = ME.utils.sparse_collate(**ema_input_dict)
    # remap segment labels from zero
    input_dict["segment2label"] = []
    for i in range(len(input_dict["labels"])):
        # TODO BIGGER CHANGE CHECK!!!
        _, ret_index, ret_inv = np.unique(input_dict["labels"][i][:, -1], return_index=True,
                                          return_inverse=True)
        input_dict["labels"][i][:, -1] = torch.from_numpy(ret_inv)
        # [4, 4, 0, 0, 0, 1, 2, 3] length = num_points_in_a_pcd
        input_dict["segment2label"].append(input_dict["labels"][i][ret_index][:, :-1])
        # [4, 0, 1, 2, 3] length = num_segments_in_a_pcd
    ema_input_dict["segment2label"] = []
    for i in range(len(ema_input_dict["labels"])):
        # TODO BIGGER CHANGE CHECK!!!
        _, ret_index, ret_inv = np.unique(ema_input_dict["labels"][i][:, -1], return_index=True,
                                          return_inverse=True)
        ema_input_dict["labels"][i][:, -1] = torch.from_numpy(ret_inv)
        # [4, 4, 0, 0, 0, 1, 2, 3] length = num_points_in_a_pcd
        ema_input_dict["segment2label"].append(ema_input_dict["labels"][i][ret_index][:, :-1])
        # [4, 0, 1, 2, 3] length = num_segments_in_a_pcd

    # 3.prepare target mask
    list_labels = input_dict["labels"]  # list:[bs x torch.tensor([num_voxels_per_pcd x 2])]
    target = get_masks(list_labels,
                                list_segments=input_dict["segment2label"],
                                ignore_class_threshold=ignore_class_threshold,
                                filter_out_classes=filter_out_classes,
                                )
    for i in range(len(target)):
        target[i]["point2segment"] = input_dict["labels"][i][:, 2]
        target[i]["ema_point2segment"] = ema_input_dict["labels"][i][:, 2]
    target_full = get_masks([torch.from_numpy(l) for l in original_labels],
                                     ignore_class_threshold=ignore_class_threshold,
                                     filter_out_classes=filter_out_classes,
                                     )
    for i in range(len(target_full)):
        target_full[i]["point2segment"] = torch.from_numpy(original_labels[i][:, 2]).long()

    return (
        NoGpu(coordinates, features, original_labels, inverse_maps, full_res_coords,
              target_full, original_colors, original_normals, original_coordinates, idx,
              ema_coordinates, ema_features, ema_inverse_maps, unique_maps
              ),
        target,
        [sample[3] for sample in batch]
    )


def get_masks(list_labels, list_segments=None, ignore_class_threshold=100, filter_out_classes=[]):
    """
    :param list_labels: a list of labels, [bs x torch.tensor(n x 3)]
    :param list_segments: indicate whether create segment-level target
    :param ignore_class_threshold: when number of points is lower than it, ignore this mask
    :param filter_out_classes: what classes should be excluded
    :return:
    target, list:[bs x dict]
                target.append({
                    'labels': l,                    # torch,tensor([num_instance x 1])
                    'masks': masks,                 # torch.tensor([num_instance x num_voxels_in_pcd])
                    'segment_mask': segment_masks   # torch.tensor([num_segments x num_voxels_in_pcd])
                })
    """
    target = []

    for batch_id in range(len(list_labels)):
        label_ids = []  # all instance classes in a scene, torch.tensor([num_instance x 1])
        masks = []  # all instance masks in a scene, torch.tensor([num_instance x num_voxels_in_pcd])
        segment_masks = []
    
        # create all masks (include thing masks and stuff masks)
        instance_ids = list_labels[batch_id][:, 1].unique()
        for instance_id in instance_ids:
            if instance_id == -1:  # filter out points of [empty]
                continue
            tmp = list_labels[batch_id][list_labels[batch_id][:, 1] == instance_id]  # instance masks [n x 3]
            class_of_instance = tmp[0, 0]
            # TODO: for now we do not filter out [1:wall, 2:floor]
            if class_of_instance in filter_out_classes:
                continue
            if 255 in filter_out_classes and class_of_instance.item() == 255 and tmp.shape[0] < ignore_class_threshold:
                continue
            label_ids.append(class_of_instance)
            masks.append(list_labels[batch_id][:, 1] == instance_id)
            if list_segments:
                segment_mask = torch.zeros(list_segments[batch_id].shape[0]).bool()
                segment_mask[list_labels[batch_id][list_labels[batch_id][:, 1] == instance_id][:, 2].unique()] = True
                # segment_masks, [num_segments x torch.tensor([num_voxels_in_pcd])]
                segment_masks.append(segment_mask)
        # in case of no instance in a scene
        if len(label_ids) == 0:
            return list()
        label_ids = torch.stack(label_ids)  # torch.tensor([num_instance x 1])
        masks = torch.stack(masks)  # torch.tensor([num_instance x num_voxels_in_pcd])
        if list_segments:
            segment_masks = torch.stack(segment_masks)

        # create semantic masks
        semantic_ids = [i for i in range(NUM_CLASSES)]
        sem_label_ids = []
        sem_masks = []
        sem_segment_masks = []
        for semantic_id in semantic_ids:
            sem_mask = list_labels[batch_id][:, 0] == semantic_id
            if semantic_id in filter_out_classes:
                continue
            if 255 in filter_out_classes and semantic_id == 255 and sum(sem_mask) < ignore_class_threshold:
                continue
            sem_label_ids.append(torch.tensor(semantic_id))
            sem_masks.append(sem_mask)
            if list_segments:
                sem_segment_mask = torch.zeros(list_segments[batch_id].shape[0]).bool()
                if sum(sem_mask)>0:
                    sem_segment_mask[list_labels[batch_id][list_labels[batch_id][:, 0] == semantic_id][:, 2].unique()] = True
                # segment_masks, [num_segments x torch.tensor([num_voxels_in_pcd])]
                sem_segment_masks.append(sem_segment_mask)
        sem_label_ids = torch.stack(sem_label_ids)
        sem_masks = torch.stack(sem_masks)
        if list_segments:
            sem_segment_masks = torch.stack(sem_segment_masks)
        
        # # create semantic masks
        # new_label_ids = []
        # new_masks = []
        # new_segment_masks = []
        # # For each class of instances, find all the instance masks and combine them into a semantic mask
        # for label_id in label_ids.unique():
        #     masking = (label_ids == label_id)
        #     new_label_ids.append(label_id)
        #     new_masks.append(masks[masking, :].sum(dim=0).bool())
        #     if list_segments:
        #         new_segment_masks.append(segment_masks[masking, :].sum(dim=0).bool())
        # sem_label_ids = torch.stack(new_label_ids)
        # sem_masks = torch.stack(new_masks)
        # if list_segments:
        #     sem_segment_masks = torch.stack(new_segment_masks)

        # combine obj masks and stuff masks to get pnoptic masks
        stuff_masking = torch.logical_or(label_ids == 0, label_ids == 1)
        obj_masks = masks[~stuff_masking]
        obj_label_ids = label_ids[~stuff_masking]
        if list_segments:
            obj_segment_masks = segment_masks[~stuff_masking]
        
        # stuff_masking = torch.logical_or(sem_label_ids == 0, sem_label_ids == 1)
        # stuff_masks = sem_masks[stuff_masking]
        # stuff_label_ids = sem_label_ids[stuff_masking]
        # if list_segments:
        #     stuff_segment_masks = sem_segment_masks[stuff_masking]
        # [thing_masks, stuff_masks]
        # masks = torch.cat([obj_masks, stuff_masks])
        # label_ids = torch.cat([obj_label_ids, stuff_label_ids])
        # if list_segments:
        #     segment_masks = torch.cat([obj_segment_masks, stuff_segment_masks])
        
        #[thing_masks, semantic_masks]
        masks = torch.cat([obj_masks, sem_masks])
        label_ids = torch.cat([obj_label_ids, sem_label_ids])
        if list_segments:
            segment_masks = torch.cat([obj_segment_masks, sem_segment_masks])

        if list_segments:
            # target, list:[bs x dict]
            target.append({
                # 'ins_masks': obj_masks,  # torch.tensor([num_instance x num_voxels_in_pcd])
                # 'ins_labels': obj_label_ids,  # torch,tensor([num_instance x 1])
                # 'sem_masks': sem_masks,  # torch.tensor([num_classes x num_voxels_in_pcd])
                # 'sem_labels': sem_label_ids,  # torch,tensor([num_classes x 1])
                # 'ins_segment_mask': segment_masks,  # torch.tensor([num_segments x num_voxels_in_pcd])
                # 'sem_segment_mask': sem_segment_masks,  # torch.tensor([num_segments x num_voxels_in_pcd])
                'masks': masks,
                'labels': label_ids,
                'segment_mask': segment_masks
                
            })
        else:
            target.append({
                # 'ins_masks': obj_masks,  # torch.tensor([num_instance x num_voxels_in_pcd])
                # 'ins_labels': obj_label_ids,  # torch,tensor([num_instance x 1])
                # 'sem_masks': sem_masks,  # torch.tensor([num_classes x num_voxels_in_pcd])
                # 'sem_labels': sem_label_ids,  # torch,tensor([num_classes x 1])
                'masks': masks,
                'labels': label_ids,
            })
    return target

from random import random
import scipy
def augmentation(coordinates):
    # 2.flip
    coordinates = flip(coordinates)
    # 3.elastic_distortion
    # if random() < 0.95:
    #     for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
    #         coordinates = elastic_distortion(coordinates, granularity, magnitude)
    return coordinates
    
def flip(coordinates):
    for i in (0, 1):
        if random() < 1.:
            coord_max = np.max(coordinates[:, i])
            coordinates[:, i] = coord_max - coordinates[:, i]
        return coordinates

def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.
    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0
    )
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud

class NoGpu:
    def __init__(
            self, coordinates, features, original_labels=None, inverse_maps=None, full_res_coords=None,
            target_full=None, original_colors=None, original_normals=None, original_coordinates=None,
            idx=None, ema_coordinates = None, ema_features = None, ema_inverse_maps=None, unique_maps = None
    ):
        """ helper class to prevent gpu loading on lightning """
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps
        self.full_res_coords = full_res_coords
        self.target_full = target_full
        self.original_colors = original_colors
        self.original_normals = original_normals
        self.original_coordinates = original_coordinates
        self.idx = idx
        self.ema_coordinates = ema_coordinates
        self.ema_features = ema_features
        self.ema_inverse_maps = ema_inverse_maps
        self.unique_maps = unique_maps

    def __add__(self, other):
        labeled_bs = len(self.original_colors)
        other.coordinates[:,0] += labeled_bs
        other.ema_coordinates[:,0] += labeled_bs
        if isinstance(other, NoGpu):
            
            return NoGpu(
                         torch.cat([self.coordinates, other.coordinates]),
                         torch.cat([self.features, other.features]),
                         self.original_labels + other.original_labels,
                         self.inverse_maps + other.inverse_maps,
                         self.full_res_coords + other.full_res_coords,
                         self.target_full + other.target_full,
                         self.original_colors + other.original_colors,
                         self.original_normals + other.original_normals,
                         self.original_coordinates + other.original_coordinates,
                         self.idx,
                         torch.cat([self.ema_coordinates, other.ema_coordinates]),
                         torch.cat([self.ema_features, other.ema_features]),
                         self.ema_inverse_maps + other.ema_inverse_maps,
                         self.unique_maps + other.unique_maps,
                         )
        else:
            raise TypeError("Unsupported operand type. Must be an instance of MyClass")