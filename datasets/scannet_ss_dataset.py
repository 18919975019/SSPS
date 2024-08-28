import logging
from itertools import product
from pathlib import Path
from random import random, sample, uniform
from typing import List, Optional, Tuple, Union
from random import choice
from copy import deepcopy
from random import randrange
import torch
from random_cuboid import RandomCuboid
import albumentations as A
import numpy as np
import scipy
import volumentations as V
import yaml
from torch.utils.data import Dataset
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}


def random_around_points(
        coordinates, color, normals, labels, rate=0.2, noise_rate=0, ignore_label=255
):
    coord_indexes = sample(
        list(range(len(coordinates))), k=int(len(coordinates) * rate)
    )
    noisy_coordinates = deepcopy(coordinates[coord_indexes])
    noisy_coordinates += np.random.uniform(
        -0.2 - noise_rate, 0.2 + noise_rate, size=noisy_coordinates.shape
    )

    if noise_rate > 0:
        noisy_color = np.random.randint(0, 255, size=noisy_coordinates.shape)
        noisy_normals = np.random.rand(*noisy_coordinates.shape) * 2 - 1
        noisy_labels = np.full(labels[coord_indexes].shape, ignore_label)

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))
    else:
        noisy_color = deepcopy(color[coord_indexes])
        noisy_normals = deepcopy(normals[coord_indexes])
        noisy_labels = deepcopy(labels[coord_indexes])

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))

    return coordinates, color, normals, labels

def crop(points, x_min, y_min, z_min, x_max, y_max, z_max):
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max and z_min < z_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min},"
            " x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )
        )
    inds = np.all(
        [
            (points[:, 0] >= x_min),
            (points[:, 0] < x_max),
            (points[:, 1] >= y_min),
            (points[:, 1] < y_max),
            (points[:, 2] >= z_min),
            (points[:, 2] < z_max),
        ],
        axis=0,
    )
    return inds

class ScannetDataset:
    def __init__(self,
                 data_dir, mode="train_labeled",  # data_dir indicate the path of database
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
                 volume_augmentations_path=os.path.join(parent_dir, "Mask3D/conf/augmentation/volumentations_aug.yaml"),
                 image_augmentations_path=os.path.join(parent_dir, "Mask3D/conf/augmentation/albumentations_aug.yaml"),
                 color_mean_std=os.path.join(parent_dir, "data/processed/scannet/color_mean_std.yaml"),
                 add_colors=True,
                 add_normals=False,
                 add_coordinates=True):

        self.mode = mode
        self.ignore_label = ignore_label
        self.data_dir = data_dir
        if type(data_dir) == str:
            self.data_dir = [self.data_dir]
        self.reps_per_epoch = reps_per_epoch
        # augmentation hyperparameter
        self.is_elastic_distortion = is_elastic_distortion
        self.color_drop = color_drop
        self.instance_oversampling = instance_oversampling
        self.place_around_existing = place_around_existing
        self.max_cut_region = max_cut_region
        self.point_per_cut = point_per_cut
        self.flip_in_center = flip_in_center
        self.noise_rate = noise_rate
        self.resample_points = resample_points
        self.cropping = cropping
        self.crop_min_size = crop_min_size
        self.crop_length = crop_length
        self.version1 = cropping_v1
        self.random_cuboid = RandomCuboid(self.crop_min_size,
                                          crop_length=self.crop_length,
                                          version1=self.version1)
        # feature setting
        self.color_map = SCANNET_COLOR_MAP_20
        self.color_map[255] = (255, 255, 255)
        self.add_colors = add_colors
        self.add_normals = add_normals  # if add normals into features
        self.add_coordinates = add_coordinates  # if add coordinates into features

        self._data = self.get_data()
        label_db_filepath = os.path.join(parent_dir, "data/processed/scannet/label_database.yaml")
        labels = self._load_yaml(Path(label_db_filepath))
        # if working only on classes for validation (20classes) - discard others
        self._labels = self._select_correct_labels(labels, 20)

        # create augmentation tools
        if instance_oversampling > 0:
            self.instance_data = self._load_yaml(
                Path(label_db_filepath).parent / "instance_database.yaml"
            )
        self.volume_augmentations = V.NoOp()
        if (volume_augmentations_path is not None) and (volume_augmentations_path != "none"):
            self.volume_augmentations = V.load(Path(volume_augmentations_path), data_format="yaml")
        self.image_augmentations = A.NoOp()
        if (image_augmentations_path is not None) and (image_augmentations_path != "none"):
            self.image_augmentations = A.load(Path(image_augmentations_path), data_format="yaml")
        if Path(str(color_mean_std)).exists():
            color_mean_std = self._load_yaml(color_mean_std)
            color_mean, color_std = (
                tuple(color_mean_std["mean"]),
                tuple(color_mean_std["std"]),
            )
        if add_colors:
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

        

    def get_data(self):
        # loading database files
        data = []
        for database_path in self.data_dir:
            database_path = Path(database_path)
            print("............dataset path is at...................")
            print(database_path.resolve())
            if not (database_path / f"{self.mode}_database.yaml").exists():
                print(f"generate {database_path}/{self.mode}_database.yaml first")
                exit()
            data.extend(self._load_yaml(database_path / f"{self.mode}_database.yaml"))
        return data

    def __len__(self):
        return self.reps_per_epoch * len(self._data)

    def __getitem__(self, idx: int):
        idx = idx % len(self._data)
        points = np.load(os.path.join(parent_dir, "data", self._data[idx]["filepath"]))
        coordinates, color, normals, segments, labels = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9],
            points[:, 10:12],
        )
        raw_coordinates = coordinates.copy()
        raw_color = color
        raw_normals = normals
        if not self.add_colors:
            color = np.ones((len(color), 3))
        
        # simpledata augmentation
        # if "train" in self.mode:
        #     # 2.centralize coordinates, add random offset
        #     coordinates = self.centralize_and_offset(coordinates)
        #     color = self.image_augmentation(color)
        #     # 9.random drop all the color(color of any point is set to 255) for a scene
        #     if random() < self.color_drop:
        #         color[:] = 255

        # data agumentation
        if "train" in self.mode:
            # 1.random crop the scene
            if self.cropping:
                new_idx = self.random_cuboid(coordinates, labels[:, 1], self._remap_from_zero(labels[:, 0].copy()))
                points = points[new_idx]
                coordinates = coordinates[new_idx]
                color = color[new_idx]
                normals = normals[new_idx]
                labels = labels[new_idx]
                segments = segments[new_idx]
                raw_coordinates = raw_coordinates[new_idx]
                raw_color = raw_color[new_idx]
                raw_normals = raw_normals[new_idx]
            # 2.centralize coordinates, add random offset
            coordinates = self.centralize_and_offset(coordinates)
            # 3.augment individual instance in the scene
            if self.instance_oversampling > 0.0:
                coordinates, color, normals, labels, segments = self.augment_individual_instance(
                        coordinates, color, normals, labels, segments, self.instance_oversampling)
            # 4.flip
            coordinates = self.flip(coordinates, points)
            # 5.elastic_distortion
            if random() < 0.95:
                if self.is_elastic_distortion:
                    for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                        coordinates = self.elastic_distortion(coordinates, granularity, magnitude)
            # 6.volume&image augmentation
            coordinates, color, normals, labels = self.volume_augmentation(coordinates, color, normals, labels)
            color = self.image_augmentation(color)
            # 7.random sample a box in the scene and cut off
            if self.point_per_cut != 0:
                coordinates, normals, color, labels = self.box_cut(coordinates, normals, color, labels)
            # 8.resample points around the real points
            if (self.resample_points > 0) or (self.noise_rate > 0):
                coordinates, color, normals, labels = self.resample(coordinates, color, normals, labels)
            # 9.random drop all the color(color of any point is set to 255) for a scene
            if random() < self.color_drop:
                color[:] = 255

        # prepare labels(semantic, instance, segment) and features(r,g,b,normals,x,y,z)
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])  # normalize color
        labels = self.collate_labels(labels, segments)
        features = self.collate_features(color, normals, coordinates)

        # replace the bug scene with the first scene
        scene_id = self._data[idx]['raw_filepath'].split("/")[-2]
        if scene_id in ['scene0636_00', 'scene0154_00']:
            return self.__getitem__(0)
        return coordinates, features, labels, scene_id, raw_color, raw_normals, raw_coordinates, idx

    def centralize_and_offset(self, coordinates):
        coordinates -= coordinates.mean(0)
        try:
            coordinates += np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2
        except OverflowError as err:
            print(coordinates)
            print(coordinates.shape)
            raise err
        return coordinates

    def augment_individual_instance(
            self, coordinates, color, normals, labels, segments, oversampling=1.0
    ):
        print(f"Insert instance for augmentation...")
        # self.save_point_cloud(coordinates, color, normals, labels, segments, file_path='before_aug.npy')
        max_instance = int(len(np.unique(labels[:, 1])))
        # randomly selecting half of non-zero instances
        for instance in range(0, int(max_instance * oversampling)):
            if self.place_around_existing:
                center = choice(
                    coordinates[labels[:, 1] == choice(np.unique(labels[:, 1]))]
                )
            else:
                center = np.array([uniform(-5, 5), uniform(-5, 5), uniform(-0.5, 2)])
            instance = choice(choice(self.instance_data))
            instance_file_path = f"/home/zxhong/SSPS/data/processed/scannet/instances/"+instance["instance_filepath"].split('/')[-1]
            instance = np.load(instance_file_path)
            # ins_box = np.max(instance[:,:3],axis=0)-np.min(instance[:,:3],axis=0)
            # scene_box = (np.max(coordinates[:,:3],axis=0)-np.min(coordinates[:,:3],axis=0))/5
            # scale_factor = np.mean(scene_box/ins_box)
            # print(scale_factor)
            instance[:,:3] = instance[:,:3]*0.8

            from augmentation import horizontal_insertion, vertical_insertion
            instance_cls = instance[0,10]
            print(f"Sample an instance of class {instance_cls} for augmentation")
            
            if instance_cls not in [4, 5, 6, 7, 14, 8, 11, 16, 39]:
                print(f"Not an appropriate class to insert!")
                continue
            if instance_cls in [4, 5, 6, 7, 14, 39]:
                print(f"horizontally insert...")
                instance_coords =  horizontal_insertion(instance, coordinates, labels, try_num=5)
            elif instance_cls in [8, 11, 16]:
                print(f"vertically insert...")
                instance_coords = vertical_insertion(instance, coordinates, labels, normals, try_num=5)         
            if instance_coords.size == 0:
                continue

            instance[:,:3] = instance_coords
            # instance = np.concatenate((instance_coords, instance[:, 3:]), axis=1)
           
            max_instance = max_instance + 1
            instance[:, -1] = max_instance

            
            # aug = V.Compose(
            #     [
            #         V.Scale3d(),
            #         V.RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(1, 0, 0)),
            #         V.RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(0, 1, 0)),
            #         V.RotateAroundAxis3d(rotation_limit=np.pi, axis=(0, 0, 1)),
            #     ]
            # )(
            #     points=instance[:, :3],
            #     features=instance[:, 3:6],
            #     normals=instance[:, 6:9],
            #     labels=instance[:, 9:],
            # )
            aug = {}
            aug["points"]=instance[:, :3]
            aug["features"]=instance[:, 3:6]
            aug["normals"]=instance[:, 6:9]
            aug["labels"]=instance[:, 9:]
           
            coordinates = np.concatenate((coordinates, aug["points"]))
            color = np.concatenate((color, aug["features"]))
            normals = np.concatenate((normals, aug["normals"]))
            labels = np.concatenate((labels, aug["labels"][:,1:]))
            
            num_segments = np.max(segments)
            aug["segments"] = aug["labels"][:,0]
            for i, id in enumerate(np.unique(aug["segments"])):
                aug["segments"][aug["segments"]== id] = i+num_segments+1
            segments = np.concatenate((segments, aug["segments"]))
        # self.save_point_cloud(coordinates, color, normals, labels, segments, file_path='after_aug.npy')
  
        return coordinates, color, normals, labels, segments
    
    def save_point_cloud(self, coordinates, color, normals, labels, segments, file_path='data.npy'):
        data_dict = {
            'coordinates': coordinates,
            'color': color,
            'normals': normals,
            'labels': labels,
            'segments': segments
        }
        np.save(file_path, data_dict)


    def flip(self, coordinates, points):
        # flip in center
        if self.flip_in_center:
            # moving coordinates to center
            coordinates -= coordinates.mean(0)
            aug = V.Compose(
                [
                    V.Flip3d(axis=(0, 1, 0), always_apply=True),
                    V.Flip3d(axis=(1, 0, 0), always_apply=True),
                ]
            )

            first_crop = coordinates[:, 0] > 0
            first_crop &= coordinates[:, 1] > 0
            # x -y
            second_crop = coordinates[:, 0] > 0
            second_crop &= coordinates[:, 1] < 0
            # -x y
            third_crop = coordinates[:, 0] < 0
            third_crop &= coordinates[:, 1] > 0
            # -x -y
            fourth_crop = coordinates[:, 0] < 0
            fourth_crop &= coordinates[:, 1] < 0

            if first_crop.size > 1:
                coordinates[first_crop] = aug(points=coordinates[first_crop])["points"]
            if second_crop.size > 1:
                minimum = coordinates[second_crop].min(0)
                minimum[2] = 0
                minimum[0] = 0
                coordinates[second_crop] = aug(points=coordinates[second_crop])["points"]
                coordinates[second_crop] += minimum
            if third_crop.size > 1:
                minimum = coordinates[third_crop].min(0)
                minimum[2] = 0
                minimum[1] = 0
                coordinates[third_crop] = aug(points=coordinates[third_crop])["points"]
                coordinates[third_crop] += minimum
            if fourth_crop.size > 1:
                minimum = coordinates[fourth_crop].min(0)
                minimum[2] = 0
                coordinates[fourth_crop] = aug(points=coordinates[fourth_crop])["points"]
                coordinates[fourth_crop] += minimum
        # flip along the x or y axis
        for i in (0, 1):
            if random() < 0.5:
                coord_max = np.max(points[:, i])
                coordinates[:, i] = coord_max - coordinates[:, i]
        return coordinates

    def elastic_distortion(self, pointcloud, granularity, magnitude):
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

    def volume_augmentation(self, coordinates, color, normals, labels):
        # 6.volume augmentation
        aug = self.volume_augmentations(
            points=coordinates, normals=normals, features=color, labels=labels,
        )
        coordinates, color, normals, labels = (
            aug["points"],
            aug["features"],
            aug["normals"],
            aug["labels"],
        )
        return coordinates, color, normals, labels

    def image_augmentation(self, color):
        # 7.image augmentation
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.image_augmentations(image=pseudo_image)["image"])
        return color

    def box_cut(self, coordinates, normals, color, labels):
        number_of_cuts = int(len(coordinates) / self.point_per_cut)
        for _ in range(number_of_cuts):
            size_of_cut = np.random.uniform(0.05, self.max_cut_region)
            # not wall, floor or empty
            point = choice(coordinates)
            x_min = point[0] - size_of_cut
            x_max = x_min + size_of_cut
            y_min = point[1] - size_of_cut
            y_max = y_min + size_of_cut
            z_min = point[2] - size_of_cut
            z_max = z_min + size_of_cut
            indexes = crop(
                coordinates, x_min, y_min, z_min, x_max, y_max, z_max
            )
            coordinates, normals, color, labels = (
                coordinates[~indexes],
                normals[~indexes],
                color[~indexes],
                labels[~indexes],
            )
        return coordinates, normals, color, labels

    def resample(self, coordinates, color, normals, labels,):
        coordinates, color, normals, labels = random_around_points(
                coordinates,
                color,
                normals,
                labels,
                self.resample_points,
                self.noise_rate,
                self.ignore_label,
            )
        return coordinates, color, normals, labels

    def _select_correct_labels(self, labels, num_labels):
        number_of_validation_labels = 0
        number_of_all_labels = 0
        for k, v, in labels.items():
            number_of_all_labels += 1
            if v["validation"]:
                number_of_validation_labels += 1

        if num_labels == number_of_all_labels:
            return labels
        elif num_labels == number_of_validation_labels:
            valid_labels = dict()
            for k, v, in labels.items():
                if v["validation"]:
                    valid_labels.update({k: v})
            return valid_labels
        else:
            msg = f"""not available number labels, select from:
            {number_of_validation_labels}, {number_of_all_labels}"""
            raise ValueError(msg)

    def _remap_from_zero(self, labels):
        label_list = list(self.label_info.keys())
        # print(f"label list:{label_list}")
        labels[~np.isin(labels, label_list)] = self.ignore_label
        # remap to the range from 0
        # print(f"before remap:{np.unique(labels)}")
        for i, k in enumerate(self.label_info.keys()):
            labels[labels == k] = i
        # print(f"after remap:{np.unique(labels)}")
        return labels

    def collate_labels(self, labels, segments):
        # prepare labels and map from 0 to 20(40)
        labels = labels.astype(np.int32)
        if labels.size > 0:
            labels[:, 0] = self._remap_from_zero(labels[:, 0])
        labels = np.hstack((labels, segments[..., None].astype(np.int32)))
        return labels

    def collate_features(self, color, normals, coordinates):
        # add normals and raw coordinates into features
        features = color
        if self.add_normals:
            features = np.hstack((features, normals))
        if self.add_coordinates:
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))
        return features

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            # file = yaml.load(f, Loader=Loader)
            file = yaml.load(f)
        return file
    
    @property
    def label_info(self):
        """ database file containing information labels used by dataset """
        return self._labels


if __name__ == '__main__':
    # labeled_dataset = ScannetDataset(data_dir=os.path.join(parent_dir, "data/processed/scannet"), mode="train_labeled")
    # for i in range(1):
    #     coordinates, features, labels, scene_id, raw_color, raw_normals, raw_coordinates, idx = labeled_dataset[i]
    #     print(coordinates.shape)
    #     print(features.shape)
    #     print(labels.shape)
    #     print(np.unique(labels[:,0]))
    #     print(np.unique(labels[:,1]))
    #     print(np.unique(labels[:,2]))
    #     print(scene_id)
    #     print('------------------------------')
    
    aug_labeled_dataset = ScannetDataset(instance_oversampling=0.5, data_dir=os.path.join(parent_dir, "data/processed/scannet"), mode="train_labeled")
    for i in range(1):
        coordinates, features, labels, scene_id, raw_color, raw_normals, raw_coordinates, idx = aug_labeled_dataset[i]
        print(coordinates.shape)
        print(features.shape)
        print(labels.shape)
        print(np.unique(labels[:,0]))
        print(np.unique(labels[:,1]))
        print(np.unique(labels[:,2]))
        print(scene_id)
        print('------------------------------')