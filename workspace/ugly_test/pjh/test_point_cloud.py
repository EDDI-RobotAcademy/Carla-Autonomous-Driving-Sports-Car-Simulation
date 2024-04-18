import unittest

import open3d as o3d
import numpy as np
import os


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

    def test_init(self):
        current_dir = os.getcwd()
        lidar_data_folder_path = os.path.join(current_dir, "lidar_output")
        files = os.listdir(lidar_data_folder_path)
        ply_list = [os.path.join(lidar_data_folder_path, file) for file in files if file.endswith(".ply")]
        point_cloud = [o3d.io.read_point_cloud(ply) for ply in ply_list]
        o3d.visualization.draw_geometries(point_cloud)

    def test_some(self):
        range_min_xyz = (-80, -80, 0)
        range_max_xyz = (80, 80, 80)
        x_min_val, y_min_val, z_min_val = range_min_xyz
        x_max_val, y_max_val, z_max_val = range_max_xyz
        lineset_yz, lineset_nth_yz = get_grid_lineset(z_min_val, z_max_val, y_min_val, y_max_val, 0, 1)
        lineset_zx, lineset_nth_zx = get_grid_lineset(x_min_val, x_max_val, z_min_val, z_max_val, 1, 1)
        lineset_xy, lineset_nth_xy = get_grid_lineset(y_min_val, y_max_val, x_min_val, x_max_val, 2, 1)

        def get_local_ply():
            # List of file paths for the .ply files
            current_dir = os.path.dirname(__file__)
            lidar_data_folder_path = os.path.join(current_dir, "lidar_output")
            files = os.listdir(lidar_data_folder_path)
            ply_list = [os.path.join(lidar_data_folder_path, file) for file in files if file.endswith(".ply")]

            # Read the point clouds and store them
            point_clouds = [o3d.io.read_point_cloud(ply) for ply in ply_list]
            print(point_clouds)
            return point_clouds

        # Initialize visualizer with key callbacks
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(height=1080)

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


if __name__ == '__main__':
    unittest.main()
