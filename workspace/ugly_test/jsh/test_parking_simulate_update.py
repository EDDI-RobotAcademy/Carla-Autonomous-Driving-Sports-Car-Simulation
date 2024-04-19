
import time
import random
import unittest

import carla

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

first_point = -28.0
second_point = -31.5

class TestParkingSimulateUpdate():

    def __init__(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        self.world = client.get_world()
        self.actor_list = []
        blueprint_library = self.world.get_blueprint_library()

        bp = random.choice(blueprint_library.filter('vehicle.tesla.cybertruck'))
        init_pos = carla.Transform(carla.Location(x=-43.9, y=-16.9, z=0.5), carla.Rotation(yaw=270))
        self.vehicle = self.world.spawn_actor(bp, init_pos)
        self.actor_list.append(self.vehicle)


    def move_to_init_parking(self):
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.2, brake=0.0))
            if self.vehicle.get_location().y < -29:
                break
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=-0.5, brake=0.0))
            if abs(self.vehicle.get_transform().rotation.yaw) > 90.1:
                self.vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                break
        return

    def moving_parking(self):
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.26, brake=0.0))
            if abs(self.vehicle.get_transform().rotation.yaw) < 0.1:
                self.vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                break
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0, brake=0.0))
            if self.vehicle.get_location().x > 12:
                self.vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                time.sleep(2.5)
                break
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0, brake=0.0))
        return

    def destroy(self):
        print('destroying actors')
        for actor in self.actor_list:
            actor.destroy()
        print('done.')


    def run(self):
        self.move_to_init_parking()
        self.moving_parking()
        #self.park()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    ego_vehicle = TestParkingSimulateUpdate()
    try:
        ego_vehicle.run()
    finally:
        if ego_vehicle is not None:
            ego_vehicle.destroy()


if __name__ == '__main__':
        main()