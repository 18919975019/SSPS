import numpy as np
from random import choice

def collision_check(instance, point_clouds, labels):
    def compute_bev_bbox(points):
        min_x, min_y = np.min(points[:, :2], axis=0)
        max_x, max_y = np.max(points[:, :2], axis=0)
        return min_x, min_y, max_x, max_y

    def compute_bev_iou(bbox1, bbox2):
        min_x1, min_y1, max_x1, max_y1 = bbox1
        min_x2, min_y2, max_x2, max_y2 = bbox2

        inter_min_x = max(min_x1, min_x2)
        inter_min_y = max(min_y1, min_y2)
        inter_max_x = min(max_x1, max_x2)
        inter_max_y = min(max_y1, max_y2)

        inter_area = max(0, inter_max_x - inter_min_x) * max(0, inter_max_y - inter_min_y)

        area1 = (max_x1 - min_x1) * (max_y1 - min_y1)
        area2 = (max_x2 - min_x2) * (max_y2 - min_y2)

        union_area = area1 + area2 - inter_area
        iou = inter_area / union_area
        return iou

    bbox1 = compute_bev_bbox(instance)
    bev_ious = []
    for instance_id in np.unique(labels[:, 1]):
        pc = point_clouds[labels[:,1]==instance_id]
        class_id = labels[labels[:,1]==instance_id][0,0]
        if class_id<3:
            continue
        # print(f"class: {class_id}:")
        bbox2 = compute_bev_bbox(pc)
        bev_iou = compute_bev_iou(bbox1, bbox2)
        # print(bev_iou)
        bev_ious.append(bev_iou)
    print(f"BEV IoU: {max(bev_ious)}")
    return max(bev_ious)

def vertical_collision_check(instance, point_clouds, labels):
    def compute_3d_bbox(points):
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        return min_coords, max_coords

    def compute_3d_iou(bbox1, bbox2):
        min_coords1, max_coords1 = bbox1
        min_coords2, max_coords2 = bbox2

        inter_min_coords = np.maximum(min_coords1, min_coords2)
        inter_max_coords = np.minimum(max_coords1, max_coords2)

        inter_dims = np.maximum(0, inter_max_coords - inter_min_coords)
        inter_volume = np.prod(inter_dims)

        volume1 = np.prod(max_coords1 - min_coords1)
        volume2 = np.prod(max_coords2 - min_coords2)

        union_volume = volume1 + volume2 - inter_volume
        iou = inter_volume / union_volume
        return iou

    bbox1 = compute_3d_bbox(instance)
    ious = []
    for instance_id in np.unique(labels[:, 1]):
        pc = point_clouds[labels[:,1] == instance_id]
        class_id = labels[labels[:,1] == instance_id][0, 0]
        if class_id < 3:
            continue
        # print(f"class: {class_id}:")
        bbox2 = compute_3d_bbox(pc)
        iou = compute_3d_iou(bbox1, bbox2)
        # print(iou)
        ious.append(iou)
    max_iou = max(ious) if ious else 0
    print(f"3D IoU: {max_iou}")
    return max_iou

def horizontal_insertion(instance, point_clouds, labels, try_num=10):
    for i in range(try_num):
        # random select center from the floor
        floor_mean_z = np.mean(point_clouds[labels[:,0]==2][:,2])
        center = choice(point_clouds[labels[:,0]==2][:,:3])
        center[2]=floor_mean_z
        # put the instance to the center position
        instance_coords = instance[:, :3] - instance[:, :3].mean(axis=0) # put the xy center as (0,0)
        instance_coords[:,2] -= np.min(instance_coords[:,2]) # put the z of instance from 0
        instance_coords += center
        max_bev_iou = collision_check(instance_coords, point_clouds, labels)
        if max_bev_iou<0.1:
            return instance_coords
    print("failed to find the appropriate position for augmentation...")
    return np.array([])

def vertical_insertion(instance, point_clouds, labels, normals, try_num=10):
    # def generate_pc(pc, background_pc, center):
    #     def compute_direction_vector_pca(points):
    #         from sklearn.decomposition import PCA
    #         """
    #         使用 PCA 计算点云的主方向向量
    #         """
    #         pca = PCA(n_components=3)
    #         pca.fit(points)
    #         direction_vector = pca.components_[0]  
    #         return direction_vector

    #     def rotation_matrix_from_vectors(vec1, vec2):
    #         """
    #         计算从vec1到vec2的旋转矩阵
    #         """
    #         vec1 = vec1 / np.linalg.norm(vec1)
    #         vec2 = vec2 / np.linalg.norm(vec2)
    #         cross_prod = np.cross(vec1, vec2)
    #         dot_prod = np.dot(vec1, vec2)
    #         if np.isclose(dot_prod, 1.0):
    #             return np.eye(3)  # 两个向量相同，返回单位矩阵
    #         if np.isclose(dot_prod, -1.0):
    #             # 两个向量相反，选择一个垂直向量进行旋转
    #             perp_vec = np.array([1, 0, 0]) if not np.allclose(vec1, [1, 0, 0]) else np.array([0, 1, 0])
    #             perp_vec -= perp_vec.dot(vec1) * vec1
    #             perp_vec /= np.linalg.norm(perp_vec)
    #             return np.eye(3) - 2 * np.outer(perp_vec, perp_vec)
            
    #         skew_symmetric = np.array([
    #             [0, -cross_prod[2], cross_prod[1]],
    #             [cross_prod[2], 0, -cross_prod[0]],
    #             [-cross_prod[1], cross_prod[0], 0]
    #         ])
    #         return np.eye(3) + skew_symmetric + (skew_symmetric @ skew_symmetric) * (1 / (1 + dot_prod))

    #     def attach_to_wall(instance_points, wall_points, center):
    #         """
    #         旋转和平移点云，使其与墙壁点云平行并贴合
    #         """
    #         wall_vector = compute_direction_vector_pca(wall_points)
    #         instance_vector = compute_direction_vector_pca(instance_points)
            
    #         rot_matrix = rotation_matrix_from_vectors(instance_vector, wall_vector)
            
    #         # 应用旋转矩阵到窗户点云
    #         rotated_instance_points = (rot_matrix @ instance_points.T).T

    #         # 计算中心点
    #         instance_center = np.mean(rotated_instance_points, axis=0)

    #         # 计算平移向量
    #         translation_vector = center - instance_center

    #         # 应用平移
    #         instance_pc = rotated_instance_points + translation_vector
    #         return instance_pc


    #     aligned_pc = attach_to_wall(pc, background_pc, center)
    #     return aligned_pc
    def generate_pc(instance_points, instance_normals, wall_points, wall_normals, center):
        def rotate_and_translate(wall_points, wall_normals, instance_points, instance_normals, center):
            # 估计wall点云的法线
            instance_normals = estimate_normals(instance_points, instance_normals)
            wall_normal = estimate_normals(wall_points, wall_normals)
            # 计算旋转矩阵
            rotation_matrix = rotation_matrix_from_vectors(instance_normals, wall_normal)
            # 应用旋转矩阵到instance点云
            rotated_instance_points = rotate_points(instance_points, rotation_matrix)
            # 计算平移向量
            translation_vector = calculate_translation(center, rotated_instance_points)
            # 应用平移向量
            translated_instance_points = rotated_instance_points + translation_vector

            return translated_instance_points

        def estimate_normals(points, normals):
            # 计算点云的法线
            # normals是每个点附近平面的法线，points是点云的坐标
            # covariance_matrix = np.cov(points.T)
            # eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            # normal = eigenvectors[:, np.argmin(eigenvalues)]
            return np.mean(normals,axis=0)

        def rotation_matrix_from_vectors(vec1, vec2):
            # 计算从vec1到vec2的旋转矩阵
            v = np.cross(vec1, vec2)
            s = np.linalg.norm(v)
            c = np.dot(vec1, vec2)
            skew_matrix = np.array([[0, -v[2], v[1]],
                                    [v[2], 0, -v[0]],
                                    [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + skew_matrix + np.dot(skew_matrix, skew_matrix) * ((1 - c) / (s ** 2))
            return rotation_matrix

        def rotate_points(points, rotation_matrix):
            # 应用旋转矩阵到点云
            rotated_points = np.dot(points, rotation_matrix.T)
            return rotated_points

        def calculate_translation(center, instance_points):
            # 计算平移向量，假设简单地使用两点之间的距离
            translation_vector = center - np.mean(instance_points, axis=0)
            return translation_vector
        return rotate_and_translate(wall_points, wall_normals, instance_points, instance_normals, center)
    

    for i in range(try_num):
        wall_instance_ids = np.unique(labels[labels[:,0]==1][:,1])
        if wall_instance_ids.shape[0] == 0:
            break
        wall_id = choice(wall_instance_ids)
        wall_coords = point_clouds[labels[:,1]==wall_id]
        insert_pos = choice(wall_coords)
        instance_coords = instance[:,:3]
        instance_normals = instance[:,6:9]
        wall_normals = normals[labels[:,1]==wall_id]
        instance_coords = generate_pc(instance_coords, instance_normals, wall_coords, wall_normals, insert_pos)
        max_bev_iou = vertical_collision_check(instance_coords, point_clouds, labels)
        if max_bev_iou<0.1:
            return instance_coords

    print("failed to find the appropriate position for augmentation...")
    return np.array([])
