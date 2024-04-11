import unittest

import carla
import math
import random
import numpy as np
import cv2


camera_data = {}


def rgb_callback(image, data_dict):
    data_dict['rgb_image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))


def sem_callback(image, data_dict):
    image.convert(carla.ColorConvertor.CityScapesPalette)
    data_dict['sem_image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))


def depth_callback(image, data_dict):
    image.convert(carla.ColorConvertor.LogarithmicDepth)
    data_dict['depth_image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))


class MyTestCase(unittest.TestCase):
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

    spectator = world.get_spectator()
    transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)))
    spectator.set_transform(transform)

    for v in world.get_actors().filter('vehicle.*'):
        v.set_autopilot(True)

    camera_init_trans = carla.Transform(carla.Location(x=0.4, z=1.6))

    rgb_camera_bp = bp_lib.find('sensor.camera.rgb')
    rgb_camera = world.spawn_actor(rgb_camera_bp, camera_init_trans, attach_to=vehicle)

    sem_camera_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    sem_camera = world.spawn_actor(sem_camera_bp, camera_init_trans, attach_to=vehicle)

    depth_camera_bp = bp_lib.find('sensor.camera.depth')
    depth_camera = world.spawn_actor(depth_camera_bp, camera_init_trans, attach_to=vehicle)

    image_w = rgb_camera_bp.get_attribute("image_size_x").as_int()
    image_h = rgb_camera_bp.get_attribute("image_size_y").as_int()

    camera_data = {'rgb_image': np.zeros((image_h, image_w, 4)),
                   'sem_image': np.zeros((image_h, image_w, 4)),
                   'depth_image': np.zeros((image_h, image_w, 4))}

    rgb_camera.listen(lambda image: rgb_callback(image, camera_data))
    sem_camera.listen(lambda image: sem_callback(image, camera_data))
    depth_camera.listen(lambda depth: depth_callback(depth, camera_data))

    cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)

    first_row = np.concatenate((camera_data['rgb_image'], camera_data['depth_image'], camera_data['sem_image']))

    cv2.imshow('All Cameras', first_row)
    cv2.waitKey(1)

    while True:
        cv2.imshow('All Cameras', first_row)

        if cv2.waitKey(1) == ord('q'):
            break

    rgb_camera.stop()
    sem_camera.stop()
    depth_camera.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
