import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
from collate_function import VoxelizeCollate
from scannet_ss_dataset import ScannetDataset
from torch.utils.data import DataLoader
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(parent_dir, "data/processed/scannet")

def make_dataloader(labeled_bs, unlabeled_bs, val_bs=2):
    train_labeled_dataset = ScannetDataset(
                 data_dir=DATA_DIR,
                 mode="train_labeled",  # data_dir indicate the path of database
                 ignore_label=255, reps_per_epoch=1,
                 is_elastic_distortion=True,
                 color_drop=0.0,
                 instance_oversampling=0.5,
                 place_around_existing=False,
                 max_cut_region=0,
                 point_per_cut=0,
                 flip_in_center=False,
                 noise_rate=0.0,
                 resample_points=0.0,
                 cropping=False,
                 crop_min_size=20000,
                 crop_length=6.0,
                 cropping_v1=True,
                 add_colors=True,
                 add_normals=False,
                 add_coordinates=True)

    c_fn = VoxelizeCollate(
        ignore_label=255,
        voxel_size=0.02,
        mode="train_labeled",
        small_crops=False,
        very_small_crops=False,
        batch_instance=False,
        probing=False,
        task="panoptic_segmentation",
        ignore_class_threshold=100,
        filter_out_classes=[255],
        label_offset=0,
    )
    
    LABELED_DATALOADER = DataLoader(train_labeled_dataset,
                                         batch_size=4,
                                         shuffle=True,
                                         num_workers=labeled_bs,
                                         pin_memory=False,
                                         collate_fn=c_fn
                                         )
    train_unlabeled_dataset = ScannetDataset(
                 data_dir=DATA_DIR,
                 mode="train_unlabeled",  # data_dir indicate the path of database
                 ignore_label=255, reps_per_epoch=1,
                 is_elastic_distortion=True,
                 color_drop=0.0,
                 instance_oversampling=0,
                 place_around_existing=False,
                 max_cut_region=0,
                 point_per_cut=0,
                 flip_in_center=False,
                 noise_rate=0.0,
                 resample_points=0.0,
                 cropping=False,
                 crop_min_size=20000,
                 crop_length=6.0,
                 cropping_v1=True,
                 add_colors=True,
                 add_normals=False,
                 add_coordinates=True)
    c_fn = VoxelizeCollate(
        ignore_label=255,
        voxel_size=0.02,
        mode="train_unlabeled",
        small_crops=False,
        very_small_crops=False,
        batch_instance=False,
        probing=False,
        task="panoptic_segmentation",
        ignore_class_threshold=100,
        filter_out_classes=[255],
        label_offset=0,
    )
    UNLABELED_DATALOADER = DataLoader(train_unlabeled_dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=unlabeled_bs,
                                    pin_memory=False,
                                    collate_fn=c_fn
                                    )
    validation_dataset = ScannetDataset(
                 data_dir=DATA_DIR,
                 mode="validation",  # data_dir indicate the path of database
                 ignore_label=255, reps_per_epoch=1,
                 is_elastic_distortion=True,
                 color_drop=0.0,
                 instance_oversampling=0,
                 place_around_existing=False,
                 max_cut_region=0,
                 point_per_cut=0,
                 flip_in_center=False,
                 noise_rate=0.0,
                 resample_points=0.0,
                 cropping=False,
                 crop_min_size=20000,
                 crop_length=6.0,
                 cropping_v1=True,
                 add_colors=True,
                 add_normals=False,
                 add_coordinates=True)
    c_fn = VoxelizeCollate(
        ignore_label=255,
        voxel_size=0.02,
        mode="validation",
        small_crops=False,
        very_small_crops=False,
        batch_instance=False,
        probing=False,
        task="panoptic_segmentation",
        ignore_class_threshold=100,
        filter_out_classes=[255],
        label_offset=0,
    )
    VALIDATION_DATALOADER = DataLoader(validation_dataset,
                                         batch_size=val_bs,
                                         shuffle=False,
                                         num_workers=4,
                                         pin_memory=False,
                                         collate_fn=c_fn
                                         )

    return LABELED_DATALOADER, UNLABELED_DATALOADER, VALIDATION_DATALOADER

if __name__ == '__main__':
    # labeled_dataset = ScannetDataset(data_dir=os.path.join(parent_dir, "data/processed/scannet"), mode="validation")
    # c_fn = VoxelizeCollate(
    #     ignore_label=255,
    #     voxel_size=0.02,
    #     mode="validation",
    #     small_crops=False,
    #     very_small_crops=False,
    #     batch_instance=False,
    #     probing=False,
    #     task="panoptic_segmentation",
    #     ignore_class_threshold=100,
    #     filter_out_classes=[],
    #     label_offset=0,
    # )
    # LABELED_DATALOADER = DataLoader(labeled_dataset,
    #                                      batch_size=2,
    #                                      shuffle=True,
    #                                      num_workers=2,
    #                                      pin_memory=False,
    #                                      collate_fn=c_fn
    #                                      )

    LABELED_DATALOADER, UNLABELED_DATALOADER, VALIDATION_DATALOADER = make_dataloader(1,1,1)
    all_labels = []
    all_scene_ids = []
    import tqdm
    # bar = tqdm(LABELED_DATALOADER)
    for i,batch in enumerate(LABELED_DATALOADER):
        print(f"processing {i} sample")
        data, target, scene_ids = batch
        batch_size = len(target)
        # print(data.coordinates.shape)
        # print(data.ema_coordinates.shape)
        # print(data.features.shape)
        # print(data.ema_features.shape)
        # sample_mask = data.coordinates[:,0]==0
        # coordinates = data.ema_coordinates[sample_mask][:,1:].numpy()
        # features = data.ema_features[sample_mask].numpy()
        
        # gt_masks = target[0]["masks"].numpy()
        # gt_labels = target[0]["labels"].numpy()
        # print(gt_labels)
        
        for bid in range(batch_size):
            all_labels.append(target[bid]["labels"][:-20])
        # for bid in range(batch_size):
        #     all_scene_ids.append(scene_ids[bid])
    # all_labels = torch.cat(all_labels)
    # print(all_labels)
    all_labels = torch.cat(all_labels)
    print(all_labels)
    cls_id, counts = torch.unique(all_labels, sorted=True, return_counts=True)
    print(cls_id)
    print(counts)
    # unique_elements, counts = torch.unique(all_labels, return_counts=True)
    # frequency = torch.stack((unique_elements, counts), dim=1)
    # print(frequency)
        # save_path = "training_sample.npz"
        # import numpy as np
        # np.savez(save_path, scene_id=scene_ids[0], features=features, coords=coordinates, gt_masks=gt_masks, gt_labels=gt_labels)
        # exit()
        # print(data.coordinates.shape)
        # print(data.features.shape)
        # print(data.inverse_maps[0].shape)
        # print(data.inverse_maps[1].shape)
        # print(data.full_res_coords[0].shape)
        # print(data.full_res_coords[1].shape)
        # print(data.target_full[0]["masks"].shape)
        # print(data.target_full[1]["masks"].shape)
        # print(data.original_colors[0].shape)
        # print(data.original_colors[1].shape)
        # print(data.original_labels[0].shape)
        # print(data.original_labels[1].shape)
        # print(torch.unique(data.coordinates[:,0]))
        # print(torch.unique(data.features[:,3]))
        # print(target[0].keys())

        # print(target[0]["masks"].shape)
        # print(target[0]["labels"])
        # print(target[0]["point2segment"].shape)
        # print(_)
