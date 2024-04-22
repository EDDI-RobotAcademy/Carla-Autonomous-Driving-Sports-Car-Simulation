import time
import unittest

import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
import pandas as pd


def get_local_ply():
    # List of file paths for the .ply files
    current_dir = os.path.dirname(__file__)
    lidar_data_folder_path = os.path.join(current_dir, "lidar_output")
    files = os.listdir(lidar_data_folder_path)
    ply_list = [os.path.join(lidar_data_folder_path, file) for file in files if file.endswith(".ply")]

    # Read the point clouds and store them
    point_clouds = [o3d.io.read_point_cloud(ply) for ply in ply_list]
    return point_clouds


def get_mesh():
    # List of file paths for the .ply files
    current_dir = os.path.dirname(__file__)
    lidar_data_folder_path = os.path.join(current_dir, "lidar_output")
    files = os.listdir(lidar_data_folder_path)
    ply_list = [os.path.join(lidar_data_folder_path, file) for file in files if file.endswith(".ply")]
    # mesh = [o3d.io.read_triangle_mesh(ply_list[i]) for i in range(len(ply_list) - 1)]
    mesh = o3d.io.read_triangle_mesh(ply_list[0])
    return mesh


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


class TestPointCloud(unittest.TestCase):
    current_index = 0
    point_clouds = None
    exit = False

    def test_basic_point_cloud(self):
        point_clouds = get_local_ply()
        o3d.visualization.draw_geometries(point_clouds)

    def test_detect_bounding_boxes_of_objects(self):
        point_clouds = get_local_ply()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector()
        for i in range(len(point_clouds) - 1):
            points = np.asarray(point_clouds[i].points)
            pcd.points.extend(points)

        pcd_1 = pcd.voxel_down_sample(voxel_size=0.05)
        pcd_2, inliers = pcd_1.remove_radius_outlier(nb_points=20, radius=0.3)
        plane_model, road_inliers = pcd_2.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=100)
        pcd_3 = pcd_2.select_by_index(road_inliers, invert=True)

        clusterer = HDBSCAN(min_cluster_size=30)
        clusterer.fit(np.array(pcd_3.points))
        labels = clusterer.labels_

        max_label = labels.max()
        print(f'point cloud has {max_label + 1} clusters')
        colors = plt.get_cmap("tab20")(labels / max_label if max_label > 0 else 1)
        colors[labels < 0] = 0
        pcd_3.colors = o3d.utility.Vector3dVector(colors[:, :3])

        bbox_objects = []
        indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()

        MAX_POINTS = 3000
        MIN_POINTS = 50

        for i in range(0, len(indexes)):
            nb_points = len(pcd_3.select_by_index(indexes[i]).points)
            if (nb_points > MIN_POINTS and nb_points < MAX_POINTS):
                sub_cloud = pcd_3.select_by_index(indexes[i])
                bbox_object = sub_cloud.get_axis_aligned_bounding_box()
                bbox_object.color = (0, 0, 1)
                bbox_objects.append(bbox_object)
                print("ID: {}\ncenter: {}\nbox points: {}".format(i, bbox_object.get_center(), bbox_object.get_box_points()))

        print("Number of Boundinb Box : ", len(bbox_objects))

        list_of_visuals = []
        list_of_visuals.append(pcd_3)
        list_of_visuals.extend(bbox_objects)
        o3d.visualization.draw_geometries(list_of_visuals)

    def test_bounding_box_of_field(self):
        point_clouds = get_local_ply()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector()
        for i in range(len(point_clouds) - 1):
            points = np.asarray(point_clouds[i].points)
            pcd.points.extend(points)
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)
        obb = pcd.get_oriented_bounding_box()
        obb.color = (0, 1, 0)
        print(np.asarray(obb.get_box_points()))
        coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=3, origin=np.array([0.0, 0.0, 0.0]))

        o3d.visualization.draw_geometries([pcd, aabb, obb, coord])

    def test_continuous_point_cloud_with_grid(self):
        range_min_xyz = (-80, -80, 0)
        range_max_xyz = (80, 80, 80)
        x_min_val, y_min_val, z_min_val = range_min_xyz
        x_max_val, y_max_val, z_max_val = range_max_xyz
        lineset_yz, lineset_nth_yz = get_grid_lineset(z_min_val, z_max_val, y_min_val, y_max_val, 0, 1)
        lineset_zx, lineset_nth_zx = get_grid_lineset(x_min_val, x_max_val, z_min_val, z_max_val, 1, 1)
        lineset_xy, lineset_nth_xy = get_grid_lineset(y_min_val, y_max_val, x_min_val, x_max_val, 2, 1)

        # Initialize visualizer with key callbacks
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()

        # Load point clouds
        point_clouds = get_local_ply()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector()
        points = np.asarray(point_clouds[self.current_index].points)
        pcd.points.extend(points)

        coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=3, origin=np.array([0.0, 0.0, 0.0]))

        vis.add_geometry(pcd)
        vis.add_geometry(coord)
        vis.add_geometry(lineset_yz)
        vis.add_geometry(lineset_zx)
        vis.add_geometry(lineset_xy)
        vis.add_geometry(lineset_nth_yz)
        vis.add_geometry(lineset_nth_zx)
        vis.add_geometry(lineset_nth_xy)

        # Get view control and capture initial viewpoint
        view_ctl = vis.get_view_control()

        def update_visualization(vis, index, view_ctl, viewpoint_params):
            pcd.points = o3d.utility.Vector3dVector()
            pcd.points.extend(np.asarray(point_clouds[index].points))
            vis.update_geometry(pcd)

            view_ctl.convert_from_pinhole_camera_parameters(viewpoint_params)

        def next_callback(vis):
            if self.current_index < len(point_clouds) - 1:
                # Capture current viewpoint before moving to next
                new_viewpoint_params = view_ctl.convert_to_pinhole_camera_parameters()
                self.current_index += 1
                update_visualization(vis, self.current_index, view_ctl, new_viewpoint_params)

        def previous_callback(vis):
            if self.current_index > 0:
                # Capture current viewpoint before moving to next
                new_viewpoint_params = view_ctl.convert_to_pinhole_camera_parameters()
                self.current_index -= 1
                update_visualization(vis, self.current_index, view_ctl, new_viewpoint_params)

        def quit_callback(vis):
            vis.close()  # Close the visualizer

        # Register key callbacks
        vis.register_key_callback(ord('N'), next_callback)
        vis.register_key_callback(ord('P'), previous_callback)
        vis.register_key_callback(ord('Q'), quit_callback)

        # Run the visualizer
        vis.run()
        vis.destroy_window()

    def test_point_pick(self):
        # shift + mouse left click = choose circle
        # shift + mouse right click = remove circle
        # shift +/- = change size of circle

        point_clouds = get_local_ply()
        vis = o3d.visualization.VisualizerWithEditing()
        for i, pcd in enumerate(point_clouds):
            vis.create_window(width=800, height=800)
            vis.add_geometry(pcd)
            vis.run()  # user picks points
            vis.destroy_window()
            indices = vis.get_picked_points()
            pcd_points = np.array(pcd.points)

            print("\n{}th point cloud".format(i))
            for index in indices:
                print(index, pcd_points[index])

    def test_point_cloud_in_real_time(self):
        range_min_xyz = (-80, -80, 0)
        range_max_xyz = (80, 80, 80)
        x_min_val, y_min_val, z_min_val = range_min_xyz
        x_max_val, y_max_val, z_max_val = range_max_xyz
        lineset_yz, lineset_nth_yz = get_grid_lineset(z_min_val, z_max_val, y_min_val, y_max_val, 0, 1)
        lineset_zx, lineset_nth_zx = get_grid_lineset(x_min_val, x_max_val, z_min_val, z_max_val, 1, 1)
        lineset_xy, lineset_nth_xy = get_grid_lineset(y_min_val, y_max_val, x_min_val, x_max_val, 2, 1)

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()

        self.point_clouds = get_local_ply()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector()
        pcd.points.extend(np.asarray(self.point_clouds[self.current_index].points))

        coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=3, origin=np.array([0.0, 0.0, 0.0]))

        vis.add_geometry(pcd)
        vis.add_geometry(coord)
        vis.add_geometry(lineset_yz)
        vis.add_geometry(lineset_zx)
        vis.add_geometry(lineset_xy)
        vis.add_geometry(lineset_nth_yz)
        vis.add_geometry(lineset_nth_zx)
        vis.add_geometry(lineset_nth_xy)

        def quit_callback(vis):
            vis.close()  # Close the visualizer
            self.exit = True

        vis.register_key_callback(ord('Q'), quit_callback)

        while True:
            if len(self.point_clouds) <= self.current_index or self.exit is True:
                break
            vis.poll_events()
            vis.update_renderer()
            pcd.points = o3d.utility.Vector3dVector()
            pcd.points.extend(np.asarray(self.point_clouds[self.current_index].points))
            vis.update_geometry(pcd)

            self.point_clouds = get_local_ply()
            self.current_index += 1
            print(self.current_index)

        vis.destroy_window()
        print('Real time point cloud test is done.')

    def test_ply_point_cloud(self):
        ply_point_cloud = o3d.data.PLYPointCloud()
        pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
        print(pcd)
        print(np.asarray(pcd.points))
        o3d.visualization.draw_geometries([pcd],
                                          zoom=0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up=[-0.0694, -0.9768, 0.2024])

    def test_voxel_point_cloud(self):
        ply_point_cloud = o3d.data.PLYPointCloud()
        pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
        downpcd = pcd.voxel_down_sample(voxel_size=0.05)
        o3d.visualization.draw_geometries([downpcd],
                                          zoom=0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up=[-0.0694, -0.9768, 0.2024])

    def test_plane_segment(self):
        pcd_point_cloud = o3d.data.PCDPointCloud()
        pcd = o3d.io.read_point_cloud(pcd_point_cloud.path)

        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000)

        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                          zoom=0.8,
                                          front=[-0.4999, -0.1659, -0.8499],
                                          lookat=[2.1813, 2.0619, 2.0999],
                                          up=[0.1204, -0.9852, 0.1215])


if __name__ == '__main__':
    unittest.main()
