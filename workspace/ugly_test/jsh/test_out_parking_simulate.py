#!/usr/bin/env python

import time
import random
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

class CarlaParkVehicle():
    def __init__(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        self.world = client.get_world()
        self.actor_list = []
        blueprint_library = self.world.get_blueprint_library()

        #create ego vehicle
        bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
        init_pos = carla.Transform(carla.Location(x=25, y=-37, z=0.5), carla.Rotation(yaw=90))
        self.vehicle = self.world.spawn_actor(bp, init_pos)
        self.actor_list.append(self.vehicle)

    def move_to_init_parking(self):
        time.sleep(2)
        while True:
            if self.vehicle.get_transform().rotation.yaw < 89.5:
                self.vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=True))
                break
            if self.vehicle.get_location().x < 19.0:
                self.vehicle.apply_control(
                    carla.VehicleControl(throttle=0.3, steer=-0.5, brake=0.0, reverse=False))
                continue
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.3, steer=0.5, brake=0.0))

    def find_parking_location(self):
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0, brake=0.0, reverse=False))
            if self.vehicle.get_location().y > -20:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                break
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.3, brake=0.0, reverse=False))
            if self.vehicle.get_transform().rotation.yaw > 179.5:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                break
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.15, steer=0.3, brake=0.0, reverse=False))
            if abs(self.vehicle.get_transform().rotation.yaw) < 89.5:
                print(self.vehicle.get_location())
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                break
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.15, steer=0.15, brake=0.0, reverse=False))
            if self.vehicle.get_location().x > -1.5:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                break
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.15, steer=-0.3, brake=0.0, reverse=False))
            if abs(self.vehicle.get_transform().rotation.yaw) > 89.5:
                print(self.vehicle.get_location())
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                break
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.1, steer=0.0, brake=0.0, reverse=False))
            if self.vehicle.get_location().y < -39.5:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                break

    def parking_left(self):
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.8, brake=0.0, reverse=False))
            if abs(self.vehicle.get_transform().rotation.yaw) < 60.5:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                break
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=-0.8, brake=0.0, reverse=True))
            if abs(self.vehicle.get_transform().rotation.yaw) < 0.5:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                break
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0, brake=0.0, reverse=True))
            if self.vehicle.get_location().x < -6.5:
                print(self.vehicle.get_transform())
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                time.sleep(5)
                break

    def parking_right(self):
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=-0.8, brake=0.0, reverse=False))
            if abs(self.vehicle.get_transform().rotation.yaw) > 120.5:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                break
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.8, brake=0.0, reverse=True))
            if abs(self.vehicle.get_transform().rotation.yaw) > 178.0:
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                break
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0, brake=0.0, reverse=True))
            if self.vehicle.get_location().x > 3.5:
                print(self.vehicle.get_transform())
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                time.sleep(5)
                break

    def destroy(self):
        print('destroying actors')
        for actor in self.actor_list:
            actor.destroy()
        print('done.')


    def run(self):
        """
        main loop
        """
        # wait for ros-bridge to set up CARLA world

        self.move_to_init_parking()
        self.find_parking_location()
        #self.parking_left()
        self.parking_right()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Main function
    """
    ego_vehicle = CarlaParkVehicle()
    try:
        ego_vehicle.run()
    finally:
        if ego_vehicle is not None:
            ego_vehicle.destroy()


if __name__ == '__main__':
        main()