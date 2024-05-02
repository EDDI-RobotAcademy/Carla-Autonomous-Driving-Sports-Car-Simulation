import shutil
import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
import pandas as pd


current_dir = os.path.dirname(__file__)
lidar_data_folder_path = os.path.join(current_dir, "resources/lidar_output")

MAX_POINTS = 4000
MIN_POINTS = 30
DETECTION_MIN_POINTS = 5


def get_local_ply():
    files = os.listdir(lidar_data_folder_path)
    ply_list = [os.path.join(lidar_data_folder_path, file) for file in files if file.endswith(".ply")]
    point_clouds = [o3d.io.read_point_cloud(ply) for ply in ply_list]
    return point_clouds


class EmptySpaceDetector:
    step_count = 0

    left_detection = False
    right_detection = False

    x_extent = 3
    min_bound = (-9, -1.5, 0)
    max_bound = (9, min_bound[1] + x_extent, 1.5)
    ego_vehicle_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    ego_vehicle_box.color = (0, 1, 0)

    empty_space_box = None
    empty_space_side = ''

    bbox_objects = []

    relocation_point_index = -4444

    def detection_process(self):
        try:
            point_clouds = get_local_ply()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector()
            for i in range(len(point_clouds) - 1):
                points = np.asarray(point_clouds[i].points)
                pcd.points.extend(points * [-1, 1, 1] + [0, 0, 3])

            pcd_1 = pcd.voxel_down_sample(voxel_size=0.05)
            pcd_2, inliers = pcd_1.remove_radius_outlier(nb_points=30, radius=0.3)
            plane_model, road_inliers = pcd_2.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=100)
            pcd_3 = pcd_2.select_by_index(road_inliers, invert=True)

            clusterer = HDBSCAN(min_cluster_size=20)
            clusterer.fit(np.array(pcd_3.points))
            labels = clusterer.labels_
            max_label = labels.max()
            colors = plt.get_cmap("tab20")(labels / max_label if max_label > 0 else 1)
            colors[labels < 0] = 0
            pcd_3.colors = o3d.utility.Vector3dVector(colors[:, :3])
            indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()

            def detection_loop():
                for i in range(0, len(indexes)):
                    nb_points = len(pcd_3.select_by_index(indexes[i]).points)
                    if (nb_points > MIN_POINTS and nb_points < MAX_POINTS):
                        sub_cloud = pcd_3.select_by_index(indexes[i])
                        bbox_object = sub_cloud.get_axis_aligned_bounding_box()
                        bbox_object.color = (0, 0, 1)
                        self.bbox_objects.append(bbox_object)
                        if len(self.ego_vehicle_box.get_point_indices_within_bounding_box(sub_cloud.points)) > DETECTION_MIN_POINTS:
                            for point in sub_cloud.points:
                                if point[0] < 0 and not self.left_detection:
                                    self.left_detection = True
                                if point[0] > 0 and not self.right_detection:
                                    self.right_detection = True

                if not self.left_detection and not self.right_detection:
                    empty_space_box_min_bound = (self.min_bound[0] + 0.2, self.min_bound[1] + 0.2, 0)
                    empty_space_box_max_bound = (self.max_bound[0] - 0.2, self.max_bound[1] - 0.2, 1)
                    self.empty_space_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=empty_space_box_min_bound,
                                                                               max_bound=empty_space_box_max_bound)
                    self.empty_space_box.color = (0.5, 0.3, 0.1)
                    self.relocation_point_index = round(self.empty_space_box.get_center()[1], 2)
                    self.empty_space_side = 'b'
                elif not self.left_detection and self.right_detection:
                    empty_space_box_min_bound = (self.min_bound[0] + 0.2, self.min_bound[1] + 0.2, 0)
                    empty_space_box_max_bound = (-1.5, self.max_bound[1] - 0.2, 1)
                    self.empty_space_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=empty_space_box_min_bound,
                                                                               max_bound=empty_space_box_max_bound)
                    self.empty_space_box.color = (0.5, 0.3, 0.1)
                    self.relocation_point_index = round(self.empty_space_box.get_center()[1], 2)
                    self.empty_space_side = 'l'
                elif not self.right_detection and self.left_detection:
                    empty_space_box_min_bound = (1.5, self.min_bound[1] + 0.2, 0)
                    empty_space_box_max_bound = (self.max_bound[0] - 0.2, self.max_bound[1] - 0.2, 1)
                    self.empty_space_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=empty_space_box_min_bound,
                                                                               max_bound=empty_space_box_max_bound)
                    self.empty_space_box.color = (0.5, 0.3, 0.1)
                    self.relocation_point_index = round(self.empty_space_box.get_center()[1], 2)
                    self.empty_space_side = 'r'
                elif abs(self.step_count) >= 50:
                    self.ego_vehicle_box = None
                else:

                    if self.step_count == 0:
                        self.step_count += 1
                    elif self.step_count > 0:
                        self.step_count = -(self.step_count + 1) * 1
                    elif self.step_count < 0:
                        self.step_count = -(self.step_count - 1) * 1
                    updated_min_bound = (self.min_bound[0], self.min_bound[1] + (self.step_count * self.x_extent * 0.1), self.min_bound[2])
                    updated_max_bound = (self.max_bound[0], self.max_bound[1] + (self.step_count * self.x_extent * 0.1), self.max_bound[2])
                    self.min_bound = updated_min_bound
                    self.max_bound = updated_max_bound
                    updated_detection_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=self.min_bound,
                                                                                max_bound=self.max_bound)
                    self.ego_vehicle_box = updated_detection_box
                    self.ego_vehicle_box.color = (0, 1, 0)
                    self.left_detection = False
                    self.right_detection = False
                    self.bbox_objects.clear()

                    detection_loop()

            detection_loop()

            with open('resources/data.txt', 'r+') as file:
                file.write(str(round(self.relocation_point_index, 3)) + '\n')
                file.write(str(self.empty_space_side))

        finally:
            if os.path.exists(lidar_data_folder_path):
                shutil.rmtree(lidar_data_folder_path)
            print('LiDAR data analysis is finished.')


if __name__ == '__main__':
    empty_space_detector = EmptySpaceDetector()
    empty_space_detector.detection_process()
