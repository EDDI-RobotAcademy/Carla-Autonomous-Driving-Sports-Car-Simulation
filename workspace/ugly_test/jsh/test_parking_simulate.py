#!/usr/bin/env python

"""
park.py implements a basic parking policy for autonomous cars based on geometric information
This script spawns:
    -one ego vehicle in x=61.4, y=-7.62, z=0.05
    -two vehicle inside the parking location a side of the ego vehicle respectively in
        x=60.4, y=-10.62, z=0.05 and x=47.0, y=-10.62, z=0.05
    -a camera attached to the ego vehicle for future sensor based parking policy

all the vehicles are rotated of 180' in order to be correctly in line with the street

requirements:
running localhost environment of carla
running carla-ros-bridge

tested env:
-Ubuntu 18.04.3 LTS
-UnrealEngine 4.22
-ROS melodic
-carla 0.9.7

"""
import argparse
import logging
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

parked_locations = [
    carla.Transform(carla.Location(x=3.7, y=-35.5, z=0.5), carla.Rotation(yaw=180)),
    carla.Transform(carla.Location(x=3.7, y=-29.8, z=0.5), carla.Rotation(yaw=180))
    ]

class CarlaParkVehicle():
    """
    class responsable of:
        -spawning 3 vehicles of which one ego
        -interact with ROS and carla server
        -destroy the created objects
        -execute the parking manoeuvre
    """
    def __init__(self):
        """
        construct object CarlaParkVehicle with server connection and
        ros node initiation
        """
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        self.world = client.get_world()
        self.actor_list = []
        blueprint_library = self.world.get_blueprint_library()

        #create ego vehicle
        bp = random.choice(blueprint_library.filter('vehicle.tesla.cybertruck'))
        init_pos = carla.Transform(carla.Location(x=-1.5, y=-42, z=0.5), carla.Rotation(yaw=90))
        self.vehicle = self.world.spawn_actor(bp, init_pos)
        self.actor_list.append(self.vehicle)

        #create 2 parked vehicles
        for pos in parked_locations:
            v = self.world.spawn_actor(bp, pos)
            self.actor_list.append(v)


    def move_to_init_parking(self):
        """
        function to move the ego vehicle into 'start parking' position
        """
        while self.vehicle.get_location().x > 50:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.4, brake=0.0))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(2)

    def park(self):
        """
        function enables the ego vehicle to enter the parking spot also actuating on steering wheels

        todo:
        -a step function is applied on the steering wheels it would be more
        realistic to actuate the steering
        wheels with an increasing(decreasing) function i.e steer += (-)0.1
        -above mentioned approach requires a more sofisticated approach i.e. sensor based
        -last manoeuvre of the parking procedure (go forward for completing the parking)
        is based only on temporal information, variability can be caused by server delays
        """
        while True:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.1,  steer=0.0, brake=0.0))
            if self.vehicle.get_location().y > first_point:
                self.vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=True))
                break
        while True:
             self.vehicle.apply_control(
                 carla.VehicleControl(throttle=0.1, steer=-0.852, brake=0.0, reverse=True))
             if self.vehicle.get_location().y < second_point:
                 self.vehicle.apply_control(
                     carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=True))
                 break
        while True:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.1, steer=0.3, brake=0.0))
            if self.vehicle.get_location().y > -31.0:
                self.vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=True))
                break
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.1, steer=-0.3, brake=0.0, reverse=True))
            if self.vehicle.get_location().y < -31:
                self.vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=True))
                break
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.05, steer=-0.65, brake=0.0, reverse=True))
            if abs(self.vehicle.get_transform().rotation.yaw) > 179.5:
                self.vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=True))
                break
        while True:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.05, steer=0, brake=0.0, reverse=True))
            if self.vehicle.get_location().x > 3.6:
                self.vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=True))
                time.sleep(2.0)
                break
        return

    def destroy(self):
        """
        destroy all the actors
        """
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
        self.park()


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