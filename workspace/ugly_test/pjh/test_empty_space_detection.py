import shutil
import unittest
import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
import pandas as pd
from colorama import Fore, Style


current_dir = os.path.dirname(__file__)
lidar_data_folder_path = os.path.join(current_dir, "../../resources/lidar_output")


def get_local_ply():
    # List of file paths for the .ply files
    files = os.listdir(lidar_data_folder_path)
    ply_list = [os.path.join(lidar_data_folder_path, file) for file in files if file.endswith(".ply")]

    # Read the point clouds and store them
    point_clouds = [o3d.io.read_point_cloud(ply) for ply in ply_list]
    return point_clouds


# visualize 3D geometry lines of every single unit distances
def get_grid_lineset(h_min_val, h_max_val, w_min_val, w_max_val, ignore_axis, grid_length=1, nth_line=5):
    assert (h_min_val % 2 == 0) and (h_max_val % 2 == 0) and (w_min_val % 2 == 0) and (w_max_val % 2 == 0)

    num_h_grid = int(np.round((h_max_val - h_min_val) // grid_length, -1))
    num_w_grid = int(np.round((w_max_val - w_min_val) // grid_length, -1))

    num_h_grid_mid = num_h_grid // 2
    num_w_grid_mid = num_w_grid // 2

    grid_vertexes_order = np.zeros((num_h_grid, num_w_grid)).astype(np.int16)
    grid_vertexes = []
    vertex_order_index = 0

    for h in range(num_h_grid):
        for w in range(num_w_grid):
            grid_vertexes_order[h][w] = vertex_order_index
            if ignore_axis == 0:
                grid_vertexes.append([0, grid_length * w + w_min_val, grid_length * h + h_min_val])
            elif ignore_axis == 1:
                grid_vertexes.append([grid_length * h + h_min_val, 0, grid_length * w + w_min_val])
            elif ignore_axis == 2:
                grid_vertexes.append([grid_length * w + w_min_val, grid_length * h + h_min_val, 0])
            else:
                pass
            vertex_order_index += 1

    next_h = [0, 1]
    next_w = [1, 0]
    grid_lines = []
    grid_nth_lines = []
    for h in range(num_h_grid):
        for w in range(num_w_grid):
            here_h = h
            here_w = w
            for i in range(2):
                there_h = h + next_h[i]
                there_w = w + next_w[i]
                if (0 <= there_h and there_h < num_h_grid) and (0 <= there_w and there_w < num_w_grid):
                    if ((here_h % nth_line) == 0) and ((here_w % nth_line) == 0):
                        grid_nth_lines.append(
                            [grid_vertexes_order[here_h][here_w], grid_vertexes_order[there_h][there_w]])
                    elif ((here_h % nth_line) != 0) and ((here_w % nth_line) == 0) and i == 1:
                        grid_nth_lines.append(
                            [grid_vertexes_order[here_h][here_w], grid_vertexes_order[there_h][there_w]])
                    elif ((here_h % nth_line) == 0) and ((here_w % nth_line) != 0) and i == 0:
                        grid_nth_lines.append(
                            [grid_vertexes_order[here_h][here_w], grid_vertexes_order[there_h][there_w]])
                    else:
                        grid_lines.append([grid_vertexes_order[here_h][here_w], grid_vertexes_order[there_h][there_w]])

    color = (0.8, 0.8, 0.8)
    colors = [color for i in range(len(grid_lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grid_vertexes),
        lines=o3d.utility.Vector2iVector(grid_lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    color = (255, 0, 0)
    colors = [color for i in range(len(grid_nth_lines))]
    line_nth_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grid_vertexes),
        lines=o3d.utility.Vector2iVector(grid_nth_lines),
    )
    line_nth_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set, line_nth_set


class TestEmptySpaceDetection(unittest.TestCase):
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

    def test_detection_process(self):
        try:
            point_clouds = get_local_ply()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector()
            for i in range(len(point_clouds) - 1):
                points = np.asarray(point_clouds[i].points)
                # fix left / right inversion & z axis calibration
                pcd.points.extend(points * [-1, 1, 1] + [0, 0, 3])

            # decrease the density of points
            pcd_1 = pcd.voxel_down_sample(voxel_size=0.05)
            # remove noise
            pcd_2, inliers = pcd_1.remove_radius_outlier(nb_points=30, radius=0.3)
            # find plane with given distance threshold using RANSAC and categorize the points on the plane
            plane_model, road_inliers = pcd_2.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=100)
            # choose points that are not on the plane
            pcd_3 = pcd_2.select_by_index(road_inliers, invert=True)

            # apply object detection algorithm by point cloud
            # todo: fix the hyper parameters to optimize the simulation
            clusterer = HDBSCAN(min_cluster_size=20)
            clusterer.fit(np.array(pcd_3.points))
            # labeling clusters with different color
            labels = clusterer.labels_
            max_label = labels.max()
            print(f'point cloud has {max_label + 1} clusters')
            colors = plt.get_cmap("tab20")(labels / max_label if max_label > 0 else 1)
            colors[labels < 0] = 0
            pcd_3.colors = o3d.utility.Vector3dVector(colors[:, :3])
            indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()

            MAX_POINTS = 4000
            MIN_POINTS = 30
            DETECTIOM_MIN_POINTS = 5

            # Detect empty space by recursively exploring space
            def detection_loop():
                for i in range(0, len(indexes)):
                    nb_points = len(pcd_3.select_by_index(indexes[i]).points)
                    if (nb_points > MIN_POINTS and nb_points < MAX_POINTS):
                        sub_cloud = pcd_3.select_by_index(indexes[i])
                        bbox_object = sub_cloud.get_axis_aligned_bounding_box()
                        bbox_object.color = (0, 0, 1)
                        self.bbox_objects.append(bbox_object)
                        print("ID: {}\n"
                              "center: {}\n"
                              "box points: {}"
                              .format(i, bbox_object.get_center(), np.asarray(bbox_object.get_box_points())))
                        if len(self.ego_vehicle_box.get_point_indices_within_bounding_box(sub_cloud.points)) > DETECTIOM_MIN_POINTS:
                            print(f'{Fore.RED}point cloud in ego_vehicle_box is detected!{Style.RESET_ALL}')
                            for point in sub_cloud.points:
                                if point[0] < 0 and not self.left_detection:
                                    print(f'{Fore.GREEN}The object id: {i} is located on the left side.{Style.RESET_ALL}')
                                    self.left_detection = True
                                if point[0] > 0 and not self.right_detection:
                                    print(f'{Fore.GREEN}The object id: {i} is located on the right side.{Style.RESET_ALL}')
                                    self.right_detection = True

                if not self.left_detection and not self.right_detection:
                    print(
                        f'{Fore.RED}Both side of ego vehicle is empty space! (maybe a parking lot){Style.RESET_ALL}')
                    empty_space_box_min_bound = (self.min_bound[0] + 0.2, self.min_bound[1] + 0.2, 0)
                    empty_space_box_max_bound = (self.max_bound[0] - 0.2, self.max_bound[1] - 0.2, 1)
                    self.empty_space_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=empty_space_box_min_bound,
                                                                               max_bound=empty_space_box_max_bound)
                    self.empty_space_box.color = (0.5, 0.3, 0.1)
                    self.relocation_point_index = round(self.empty_space_box.get_center()[1], 2)
                    self.empty_space_side = 'b'
                elif not self.left_detection and self.right_detection:
                    print(
                        f'{Fore.RED}Left side of ego vehicle is empty space! (maybe a parking lot){Style.RESET_ALL}')
                    empty_space_box_min_bound = (self.min_bound[0] + 0.2, self.min_bound[1] + 0.2, 0)
                    empty_space_box_max_bound = (-1.5, self.max_bound[1] - 0.2, 1)
                    self.empty_space_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=empty_space_box_min_bound,
                                                                               max_bound=empty_space_box_max_bound)
                    self.empty_space_box.color = (0.5, 0.3, 0.1)
                    self.relocation_point_index = round(self.empty_space_box.get_center()[1], 2)
                    self.empty_space_side = 'l'
                elif not self.right_detection and self.left_detection:
                    print(
                        f'{Fore.RED}Right side of ego vehicle is empty space! (maybe a parking lot){Style.RESET_ALL}')
                    empty_space_box_min_bound = (1.5, self.min_bound[1] + 0.2, 0)
                    empty_space_box_max_bound = (self.max_bound[0] - 0.2, self.max_bound[1] - 0.2, 1)
                    self.empty_space_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=empty_space_box_min_bound,
                                                                               max_bound=empty_space_box_max_bound)
                    self.empty_space_box.color = (0.5, 0.3, 0.1)
                    self.relocation_point_index = round(self.empty_space_box.get_center()[1], 2)
                    self.empty_space_side = 'r'
                elif abs(self.step_count) >= 50:
                    print(
                        f'{Fore.RED}Ego vehicle should be relocated! (no empty space in detection range){Style.RESET_ALL}')
                    self.ego_vehicle_box = None
                else:
                    print(
                        f'{Fore.RED}Ego vehicle bounding box should be relocated! (no empty space currently){Style.RESET_ALL}')
                    if self.step_count == 0:
                        self.step_count += 1
                    elif self.step_count > 0:
                        self.step_count = -(self.step_count + 1) * 1
                    elif self.step_count < 0:
                        self.step_count = -(self.step_count - 1) * 1
                    print(f'step_count: {self.step_count}')
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

            print(f"{Fore.BLUE}Number of Bounding Box : {len(self.bbox_objects)}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Relocation Point Index : {self.relocation_point_index}{Style.RESET_ALL}")

            # Record relocation point index
            with open('../../resources/data.txt', 'r+') as file:
                file.write(str(round(self.relocation_point_index, 3)) + '\n')
                file.write(str(self.empty_space_side))

            # Draw geometries to explain situation
            list_of_visuals = []
            list_of_visuals.append(pcd_3)
            list_of_visuals.extend(self.bbox_objects)

            range_min_xyz = (-80, -80, 0)
            range_max_xyz = (80, 80, 80)
            x_min_val, y_min_val, z_min_val = range_min_xyz
            x_max_val, y_max_val, z_max_val = range_max_xyz
            lineset_yz, lineset_nth_yz = get_grid_lineset(z_min_val, z_max_val, y_min_val, y_max_val, 0, 1)
            lineset_zx, lineset_nth_zx = get_grid_lineset(x_min_val, x_max_val, z_min_val, z_max_val, 1, 1)
            lineset_xy, lineset_nth_xy = get_grid_lineset(y_min_val, y_max_val, x_min_val, x_max_val, 2, 1)

            coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=3, origin=np.array([0.0, 0.0, 0.0]))
            # x = red, y = green, and z = blue

            vis = o3d.visualization.Visualizer()
            vis.create_window()

            vis.add_geometry(pcd_3)
            for j in range(len(self.bbox_objects)):
                vis.add_geometry(self.bbox_objects[j])
            vis.add_geometry(coord)
            vis.add_geometry(lineset_yz)
            vis.add_geometry(lineset_zx)
            vis.add_geometry(lineset_xy)
            vis.add_geometry(lineset_nth_yz)
            vis.add_geometry(lineset_nth_zx)
            vis.add_geometry(lineset_nth_xy)
            if self.ego_vehicle_box is not None:
                vis.add_geometry(self.ego_vehicle_box)
            if self.empty_space_box is not None:
                vis.add_geometry(self.empty_space_box)

            vis.run()
            vis.destroy_window()

        finally:
            if os.path.exists(lidar_data_folder_path):
                shutil.rmtree(lidar_data_folder_path)


if __name__ == '__main__':
    unittest.main()
