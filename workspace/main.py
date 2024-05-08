from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import time

import cv2

try:
    sys.path.append(glob.glob(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc
from carla import VehicleLightState as vls

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
from numpy import random

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_F2
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_e
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_k
    from pygame.locals import K_j
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_y
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def draw_transform(debug, trans, col=carla.Color(255, 0, 0), lt=-1):
    debug.draw_arrow(
        trans.location, trans.location + trans.get_forward_vector(),
        thickness=0.05, arrow_size=0.1, color=col, life_time=lt)


def draw_waypoint_union(debug, w0, w1, color=carla.Color(255, 0, 0), lt=5):
    debug.draw_line(
        w0.transform.location + carla.Location(z=0.25),
        w1.transform.location + carla.Location(z=0.25),
        thickness=0.1, color=color, life_time=lt, persistent_lines=False)
    debug.draw_point(w1.transform.location + carla.Location(z=0.25), 0.1, color, lt, False)


def draw_waypoint_info(debug, w, lt=5):
    w_loc = w.transform.location
    debug.draw_string(w_loc + carla.Location(z=0.5), "lane: " + str(w.lane_id), False, yellow, lt)
    debug.draw_string(w_loc + carla.Location(z=1.0), "road: " + str(w.road_id), False, blue, lt)
    debug.draw_string(w_loc + carla.Location(z=-.5), str(w.lane_change), False, red, lt)


def draw_junction(debug, junction, l_time=10):
    """Draws a junction bounding box and the initial and final waypoint of every lane."""
    # draw bounding box
    box = junction.bounding_box
    point1 = box.location + carla.Location(x=box.extent.x, y=box.extent.y, z=2)
    point2 = box.location + carla.Location(x=-box.extent.x, y=box.extent.y, z=2)
    point3 = box.location + carla.Location(x=-box.extent.x, y=-box.extent.y, z=2)
    point4 = box.location + carla.Location(x=box.extent.x, y=-box.extent.y, z=2)
    debug.draw_line(
        point1, point2,
        thickness=0.1, color=orange, life_time=l_time, persistent_lines=False)
    debug.draw_line(
        point2, point3,
        thickness=0.1, color=orange, life_time=l_time, persistent_lines=False)
    debug.draw_line(
        point3, point4,
        thickness=0.1, color=orange, life_time=l_time, persistent_lines=False)
    debug.draw_line(
        point4, point1,
        thickness=0.1, color=orange, life_time=l_time, persistent_lines=False)
    # draw junction pairs (begin-end) of every lane
    junction_w = junction.get_waypoints(carla.LaneType.Any)
    for pair_w in junction_w:
        draw_transform(debug, pair_w[0].transform, orange, l_time)
        debug.draw_point(
            pair_w[0].transform.location + carla.Location(z=0.75), 0.1, orange, l_time, False)
        draw_transform(debug, pair_w[1].transform, orange, l_time)
        debug.draw_point(
            pair_w[1].transform.location + carla.Location(z=0.75), 0.1, orange, l_time, False)
        debug.draw_line(
            pair_w[0].transform.location + carla.Location(z=0.75),
            pair_w[1].transform.location + carla.Location(z=0.75), 0.1, white, l_time, False)


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.parked_vehicles = []
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.los_sensor = None
        self.lidar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

        self.parking_control = None
        self.parking_break = False
        self.parking_side = ''
        self.parking_relocation_position = 0

        self.parking_left = None
        self.parking_right = None

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        print(self.player.bounding_box.location)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def toggle_los(self):
        if self.los_sensor is None:
            self.los_sensor = LineOfSightSensor(self.player, self.hud)
            self.hud.notification('LineOfSightSensor On')
        elif self.los_sensor.sensor is not None:
            self.los_sensor.sensor.destroy()
            self.los_sensor = None
            self.hud.notification('LineOfSightSensor Off')

    def toggle_lidar(self):
        if self.lidar_sensor is None:
            self.lidar_sensor = LidarSensor(self.player, self.hud)
            self.hud.notification('LiDAR On')
        elif self.lidar_sensor.sensor is not None:
            self.lidar_sensor.sensor.destroy()
            self.lidar_sensor = None
            self.hud.notification('LiDAR Off')

    def toggle_parked_vehicles(self):
        if not self.parked_vehicles:
            parked_locations = [
                carla.Transform(carla.Location(x=3.7, y=-35.5, z=0.5), carla.Rotation(yaw=180)),
                carla.Transform(carla.Location(x=3.7, y=-32.7, z=0.5), carla.Rotation(yaw=180)),
                carla.Transform(carla.Location(x=3.7, y=-29.9, z=0.5), carla.Rotation(yaw=180)),
                carla.Transform(carla.Location(x=3.7, y=-27.1, z=0.5), carla.Rotation(yaw=180)),
                carla.Transform(carla.Location(x=-7, y=-38.3, z=0.5)),
                carla.Transform(carla.Location(x=-7, y=-29.9, z=0.5)),
                carla.Transform(carla.Location(x=-7, y=-27.1, z=0.5)),
            ]
            blueprint = random.choice(self.world.get_blueprint_library().filter('vehicle.tesla.model3'))
            for location in parked_locations:
                v = self.world.try_spawn_actor(blueprint, location)
                if v is not None:
                    self.parked_vehicles.append(v)
        else:
            for v in self.parked_vehicles:
                v.destroy()
                self.parked_vehicles = []

    def fix_ego_vehicle_location_to_adjacent_empty_space(self):
        self.hud.notification('Fixing ego vehicle location to adjacent empty space')
        # moving north direction decreases y value
        relocation_index = 0
        reverse = False
        current_location_y = self.player.get_location().y
        target_location_y = 0

        # analyze direction based on data
        try:
            with open('resources/data.txt', 'r+') as f:
                lines = f.readlines()
                if lines and len(lines) == 2:
                    relocation_index = float(lines[0])
                    self.parking_side = lines[1]
                    if relocation_index < 0:
                        reverse = True
                    f.seek(0)
                    f.truncate()
                else:
                    print('No data!')
                    return
        except FileNotFoundError:
            return

        # check compass and move
        compass = self.imu_sensor.compass
        if compass > 355 or compass < 5:  # north
            target_location_y = current_location_y - relocation_index
        elif 175 < compass < 185:  # south
            target_location_y = current_location_y + relocation_index
        else:
            print('The player is too tilted!')
            return

        if int(relocation_index) == -4444:
            print('The player should move position!')
            return

        control = carla.VehicleControl(throttle=0.3, steer=0.0, reverse=reverse)
        self.parking_control = control
        self.parking_relocation_position = target_location_y

        print('moving to {}'.format(self.parking_relocation_position))


    def move_to_line(self, move_distance):

        current_location_y = self.player.get_location().y
        compass = self.imu_sensor.compass

        if compass > 355 or compass < 5:  # north
            target_location_y = current_location_y - move_distance

        elif 175 < compass < 185:  # south
            target_location_y = current_location_y + move_distance

        else:
            print('The player is too tilted!')
            return

        if int(move_distance) == -4444:
            print('The player should move position!')
            return

        self.parking_relocation_position = target_location_y
        print('parking relocation position: ', self.parking_relocation_position)


    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.parked_vehicles:
            for vehicle in self.parked_vehicles:
                vehicle.destroy()
        if self.player is not None:
            self.player.destroy()

    def get_result_parking_side(self):
        return self.parking_side


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._carsim_enabled = False
        self._carsim_road = False
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_o:
                    world.toggle_los()
                elif event.key == K_y:
                    world.toggle_lidar()
                elif event.key == K_e:
                    world.toggle_parked_vehicles()
                elif event.key == K_F2:
                    world.fix_ego_vehicle_location_to_adjacent_empty_space()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("/home/melodic/.config/Epic/CarlaUE4/Saved/test1.log")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_k and (pygame.key.get_mods() & KMOD_CTRL):
                    print("k pressed")
                    world.player.enable_carsim("d:/CVC/carsim/DataUE4/ue4simfile.sim")
                elif event.key == K_j and (pygame.key.get_mods() & KMOD_CTRL):
                    self._carsim_road = not self._carsim_road
                    world.player.use_carsim_road(self._carsim_road)
                    print("j pressed, using carsim road =", self._carsim_road)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        # self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        bb = world.player.bounding_box
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Size: % 20s' % ('(x=%3.2f, y=%3.2f, z=%3.2f)' % (bb.extent.x, bb.extent.y, bb.extent.z)),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f, % 5.1f)' % (t.location.x, t.location.y, t.location.z)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        # self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(100))
        bp.set_attribute('vertical_fov', str(20))
        bp.set_attribute('range', str(5))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.8, z=1.0),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_data), 4))

        print(points)

        # code convert array into list and measure distance
        L = []
        pointslist = points.tolist()
        for i in range(len(pointslist)):
            L.append(pointslist[i - 1][-1])

        ave = sum(L) / len(L)
        print(ave)

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            print("Depth:", detect.depth)
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- LineOfSightSensor -------------------------------------------------------------
# ==============================================================================


class LineOfSightSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        self._event_count = 0
        self.sensor_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=0))  # Put this sensor on the windshield of the car.
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.obstacle')
        bp.set_attribute('distance', '10')
        bp.set_attribute('hit_radius', '10')
        bp.set_attribute('only_dynamics', 'False')
        # bp.set_attribute('debug_linetrace', 'true')
        # bp.set_attribute('sensor_tick', '1')
        # self.sensor = world.spawn_actor(bp, self.sensor_transform, attach_to=self._parent)
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LineOfSightSensor._on_LOS(weak_self, event))

    @staticmethod
    def _on_LOS(weak_self, event):
        self = weak_self()
        if not self:
            return
        # print (str(event.other_actor))
        # if event.other_actor.type_id.startswith('vehicle.'):
        #     # print ("Event %s, in line of sight with %s at distance %u" % (self._event_count, event.other_actor.type_id, event.distance))
        #     self._event_count += 1

        if event.other_actor.type_id.startswith('static.vehicle') and event.distance < 1:
            for item in self._history:
                if item[0] == event.other_actor.id:
                    return

            self._history.append((event.other_actor.id, event.other_actor.get_location()))

            print("Event %s, too close with parked %s (less than 1 meter)" % (self._event_count, event.other_actor.type_id))
            obstacle_location = event.other_actor.get_location()
            obstacle_id = event.other_actor.id
            x = obstacle_location.x
            y = obstacle_location.y
            z = obstacle_location.z
            print("id = %u, location = x: %f, y: %f, z: %f" % (obstacle_id, x, y, z))

            self._event_count += 1

# ==============================================================================
# -- LidarSensor -------------------------------------------------------------
# ==============================================================================


class LidarSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        self._event_count = 0
        self.sensor_transform = carla.Transform(carla.Location(z=3), carla.Rotation(yaw=-90.0))
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('channels', '32')
        bp.set_attribute('points_per_second', '96000')
        bp.set_attribute('rotation_frequency', '30')
        bp.set_attribute('range', '15')
        self.sensor = world.spawn_actor(bp, self.sensor_transform, attach_to=self._parent)
        self.sensor.listen(lambda point_cloud: point_cloud.save_to_disk('resources/lidar_output/%.6d.ply' % point_cloud.frame))


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-10.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.Rigid),
            # (carla.Transform(carla.Location(z=3), carla.Rotation(yaw=-90.0)), Attachment.Rigid),
            # (carla.Transform(carla.Location(y=-1.3, z=1), carla.Rotation(pitch=15.0, yaw=-90.0)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=0.0, z=30.0), carla.Rotation(pitch=-90.0)), Attachment.Rigid),
            (carla.Transform(carla.Location(z=8.0), carla.Rotation(pitch=-90.0)), Attachment.Rigid),
            # (carla.Transform(carla.Location(x=-3, y=-bound_y, z=0.5)), Attachment.Rigid),
            # (carla.Transform(carla.Location(x=-3, y=bound_y, z=0.5)), Attachment.Rigid),
            # (carla.Transform(carla.Location(x=bound_x, y=2, z=0.5), carla.Rotation(yaw=-180.0)), Attachment.Rigid),
            # (carla.Transform(carla.Location(x=bound_x, y=-2, z=0.5), carla.Rotation(yaw=-180.0)), Attachment.Rigid),
        ]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None
        self.sensor_list = []
        self.sensor_data = {}

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_camera_with_option(self, transform_index, sensor_index):
        self.transform_index = transform_index
        self.index = sensor_index
        self.toggle_camera()

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            global Camera_image
            Camera_image = array.copy()

        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

    def setup_multiple_sensors(self, sensor_index_list, transform_index=0):
        # only for rgb, depth, semantic -> index : 0, 2, 5
        rigid = carla.AttachmentType.Rigid
        bp_lib = self._parent.get_world().get_blueprint_library()
        camera_init_transform = self._camera_transforms[transform_index][0]
        weak_self = weakref.ref(self)

        for sensor_index in sensor_index_list:
            camera_bp = bp_lib.find(self.sensors[sensor_index][0])
            camera = self._parent.get_world().spawn_actor(
                camera_bp, camera_init_transform, attach_to=self._parent, attachment_type=rigid)
            self.sensor_list.append(camera)
            image_w = camera_bp.get_attribute("image_size_x").as_int()
            image_h = camera_bp.get_attribute("image_size_y").as_int()
            self.sensor_data[self.sensors[sensor_index][0]] = np.zeros((image_h, image_w, 4))
            camera.listen(lambda image: weak_self().sensor_callback(weak_self, image, sensor_index))

        for sensor in self.sensor_list:
            if sensor.is_listening:
                print('Sensor is listening')

    @staticmethod
    def sensor_callback(weak_self, image, sensor_index):
        self = weak_self()
        image.convert(self.sensors[sensor_index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.sensor_data[self.sensors[sensor_index][0]] = array

    def show_tiled_view(self):
        for sensor in self.sensor_data.keys():
            cv2.imshow(str(sensor), self.sensor_data[sensor])


# ==============================================================================
# -- LaneDetector ---------------------------------------------------------------
# ==============================================================================
class LaneDetector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self, camera_image, result_parking_side):
        if not hasattr(self, 'initialized'):
            self.camera_image = camera_image

            # manage detected lines
            self.left_space_line = []
            self.right_space_line = []

            # LiDAR result(l/b/r)
            self.result_parking_side = result_parking_side

            # line detection result(T/F)
            self.line_detection_result = None
            # move distance
            self.result_move_distance = None
            # T/F
            self.move_distance_control = True
            self.initialized = True

    def left_line_points(self, image, lines):
        left_line1_point, left_line2_point = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]


            if 0 < x1 < image.shape[1] and 0 < x2 < image.shape[1]:
                if 0 < y1 < image.shape[0]/2 and 0 < y2 < image.shape[0]/2:
                    left_line1_point.append(np.array([x1, y1, x2, y2]))

                elif image.shape[0] * 0.4 < y1 < image.shape[0] and image.shape[0] * 0.4 < y2 < image.shape[0]:
                    left_line2_point.append(np.array([x1, y1, x2, y2]))

        if left_line1_point and left_line2_point:
            left_line1_max = np.max(left_line1_point, axis=0)
            left_line2_max = np.max(left_line2_point, axis=0)

            left_line1_min = np.min(left_line1_point, axis=0)
            left_line2_min = np.min(left_line2_point, axis=0)

            # adjust to fit in the middle of the line
            left_line1_y1_average = (left_line1_max[1] + left_line1_min[1])/2
            left_line1_y2_average = (left_line1_max[3] + left_line1_min[3])/2

            left_line2_y1_average = (left_line2_max[1] + left_line2_min[1]) / 2
            left_line2_y2_average = (left_line2_max[3] + left_line2_min[3]) / 2

            left_line1 = np.array([left_line1_min[0], left_line1_y1_average, left_line1_max[2], left_line1_y2_average])
            left_line2 = np.array([left_line2_min[0], left_line2_y1_average, left_line2_max[2], left_line2_y2_average])

            self.left_space_line = np.array([left_line1, left_line2])

            return np.array([left_line1, left_line2])


        elif left_line1_point and not left_line2_point:
            left_line1_max = np.max(left_line1_point, axis=0)
            left_line1_min = np.min(left_line1_point, axis=0)

            left_line1_y1_average = (left_line1_max[1] + left_line1_min[1]) / 2
            left_line1_y2_average = (left_line1_max[3] + left_line1_min[3]) / 2

            left_line1 = np.array([left_line1_min[0], left_line1_y1_average, left_line1_max[2], left_line1_y2_average])
            self.left_space_line = np.array([left_line1])

            return np.array([left_line1])

        elif not left_line1_point and left_line2_point:
            left_line2_max = np.max(left_line2_point, axis=0)
            left_line2_min = np.min(left_line2_point, axis=0)

            left_line2_y1_average = (left_line2_max[1] + left_line2_min[1]) / 2
            left_line2_y2_average = (left_line2_max[3] + left_line2_min[3]) / 2

            left_line2 = np.array([left_line2_min[0], left_line2_y1_average, left_line2_max[2], left_line2_y2_average])
            self.left_space_line = np.array([left_line2])

            return np.array([left_line2])


    def right_line_points(self, image, lines):
        right_line1_point, right_line2_point = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if 0 < x1 < image.shape[1] and 0 < x2 < image.shape[1]:
                if 0 < y1 < image.shape[0]/2 and 0 < y2 < image.shape[0]/2:
                    right_line1_point.append(np.array([x1, y1, x2, y2]))

                elif image.shape[0] * 0.4 < y1 < image.shape[0] and image.shape[0] * 0.4 < y2 < image.shape[0]:
                    right_line2_point.append(np.array([x1, y1, x2, y2]))

        if right_line1_point and right_line2_point:
            right_line1_max = np.max(right_line1_point, axis=0)
            right_line2_max = np.max(right_line2_point, axis=0)

            right_line1_min = np.min(right_line1_point, axis=0)
            right_line2_min = np.min(right_line2_point, axis=0)

            right_line1_y1_average = (right_line1_max[1] + right_line1_min[1]) / 2
            right_line1_y2_average = (right_line1_max[3] + right_line1_min[3]) / 2

            right_line2_y1_average = (right_line2_max[1] + right_line2_min[1]) / 2
            right_line2_y2_average = (right_line2_max[3] + right_line2_min[3]) / 2

            right_line1 = np.array([right_line1_min[0], right_line1_y1_average, right_line1_max[2], right_line1_y2_average])
            right_line2 = np.array([right_line2_min[0], right_line2_y1_average, right_line2_max[2], right_line2_y2_average])

            self.right_space_line = np.array([right_line1, right_line2])

            return np.array([right_line1, right_line2])


        elif right_line1_point and not right_line2_point:
            right_line1_max = np.max(right_line1_point, axis=0)
            right_line1_min = np.min(right_line1_point, axis=0)

            right_line1_y1_average = (right_line1_max[1] + right_line1_min[1]) / 2
            right_line1_y2_average = (right_line1_max[3] + right_line1_min[3]) / 2

            right_line1 = np.array([right_line1_min[0], right_line1_y1_average, right_line1_max[2], right_line1_y2_average])
            self.right_space_line = np.array([right_line1])

            return np.array([right_line1])


        elif not right_line1_point and right_line2_point:
            right_line2_max = np.max(right_line2_point, axis=0)
            right_line2_min = np.min(right_line2_point, axis=0)

            right_line2_y1_average = (right_line2_max[1] + right_line2_min[1]) / 2
            right_line2_y2_average = (right_line2_max[3] + right_line2_min[3]) / 2

            right_line2 = np.array([right_line2_min[0], right_line2_y1_average, right_line2_max[2], right_line2_y2_average])
            self.right_space_line = np.array([right_line2])

            return np.array([right_line2])


    def lines_distance_ratio(self, image, line):

        if len(line) == 2:
            image_height = image.shape[0]
            distance_lines = line[1][1] - line[0][1]
            distance_ratio = distance_lines / float(image_height)
            return distance_ratio

        else:
            return 0


    def parking_space_detection(self, distance_ratio):
        if distance_ratio > 0.29:
            print("Parking available!")
            # print(self.left_space_line)
            # print(self.right_space_line)
            self.line_detection_result = True

        else:
            # print("No parking available!")
            self.line_detection_result = False


    def move_distance(self, roi_polygons, line):
        if self.line_detection_result is True:
            if len(line) == 2:
                roi_height = int(roi_polygons.shape[0] * 0.8)
                distance_between_lines = line[1][1] - line[0][1]
                move_distance = ((2.8/distance_between_lines) * (float(roi_height) - line[1][1])) + 2
                self.result_move_distance = move_distance

                print("move distance: ", move_distance)

        else:
            self.result_move_distance = 0


    def get_move_distance(self):
        return self.result_move_distance


    def canny(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        canny_image = cv2.Canny(blurred_image, 50, 150)
        return canny_image


    def region_of_interest_left(self, image):
        height = image.shape[0]
        width = image.shape[1]

        # rectangle points
        # left low, right low, right high, left high
        polygons = np.array([
            [(int(0.1 * width), int(0.8 * height)), (int(0.4 * width), int(0.8 * height)),
             (int(0.4 * width), int(0.2 * height)), (int(0.1 * width), int(0.2 * height))]
        ])

        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image, mask)

        return masked_image


    def region_of_interest_right(self, image):
        height = image.shape[0]
        width = image.shape[1]

        polygons = np.array([
            [(int(0.6 * width), int(0.8 * height)), (int(0.9 * width), int(0.8 * height)),
             (int(0.9 * width), int(0.2 * height)), (int(0.6 * width), int(0.2 * height))]
        ])

        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image, mask)

        return masked_image



    def left_roi_image(self, image):
        height = image.shape[0]
        width = image.shape[1]

        polygons = np.array([
            [(int(0.1 * width), int(0.8 * height)), (int(0.4 * width), int(0.8 * height)),
             (int(0.4 * width), int(0.2 * height)), (int(0.1 * width), int(0.2 * height))]
        ])

        roi_image = cv2.polylines(image, [polygons], True, (0, 255, 0), 10)

        return roi_image


    def right_roi_image(self, image):
        height = image.shape[0]
        width = image.shape[1]

        polygons = np.array([
            [(int(0.6 * width), int(0.8 * height)), (int(0.9 * width), int(0.8 * height)),
             (int(0.9 * width), int(0.2 * height)), (int(0.6 * width), int(0.2 * height))]
        ])

        roi_image = cv2.polylines(image, [polygons], True, (0, 255, 0), 10)

        return roi_image


    def display_left_lines(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = map(int, line)
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

        return line_image


    def display_right_lines(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = map(int, line)
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

        return line_image


    def run_lane_detection(self):
        lane_image = np.copy(self.camera_image)
        canny_image = self.canny(lane_image)

        if self.result_parking_side == 'l' or self.result_parking_side == 'b':

            # left_line
            cropped_image_left = self.region_of_interest_left(canny_image)
            left_lines = cv2.HoughLinesP(
                cropped_image_left, 1, np.pi / 180, 15, np.array([]), minLineLength=10, maxLineGap=10)

            if left_lines is not None:
                left_line_point = self.left_line_points(cropped_image_left, left_lines)
                distance_ratio = self.lines_distance_ratio(cropped_image_left, self.left_space_line)
                self.parking_space_detection(distance_ratio)
                if self.move_distance_control:
                    self.move_distance(cropped_image_left, self.left_space_line)
                    self.move_distance_control = False


                if left_line_point is not None:
                    left_line_image = self.display_left_lines(lane_image, left_line_point)
                    combined_left_image = cv2.addWeighted(lane_image, 0.8, left_line_image, 1, 1)

                else:
                    combined_left_image = lane_image

            else:
                combined_left_image = lane_image

            cv2.imshow('combined_left_image', self.left_roi_image(combined_left_image))


        elif self.result_parking_side == 'r':
            # right_line
            cropped_image_right = self.region_of_interest_right(canny_image)
            right_lines = cv2.HoughLinesP(
                cropped_image_right, 1, np.pi / 180, 15, np.array([]), minLineLength=10, maxLineGap=10)

            if right_lines is not None:
                right_line_point = self.right_line_points(cropped_image_right, right_lines)
                distance_ratio = self.lines_distance_ratio(cropped_image_right, self.right_space_line)
                self.parking_space_detection(distance_ratio)
                if self.move_distance_control:
                    self.move_distance(cropped_image_right, self.right_space_line)
                    self.move_distance_control = False

                if right_line_point is not None:
                    right_line_image = self.display_right_lines(lane_image, right_line_point)
                    combined_right_image = cv2.addWeighted(lane_image, 0.8, right_line_image, 1, 1)

                else:
                    combined_right_image = lane_image

            else:
                combined_right_image = lane_image

            cv2.imshow('combined_right_image', self.right_roi_image(combined_right_image))

        cv2.waitKey(1)

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world, args.autopilot)

        clock = pygame.time.Clock()

        player_moving = False
        empty_space_relocation_finished = False
        camera_setting_for_line_detection = False

        move_possibility = True

        while True:
            clock.tick_busy_loop(30)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            world.render(display)

            if world.parking_control is not None:
                world.player.apply_control(world.parking_control)
                player_moving = True

            if abs(world.player.get_location().y - world.parking_relocation_position) < 0.07 and player_moving:
                world.parking_break = True
                world.parking_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=True)

            if world.parking_break and world.player.get_velocity() == carla.Vector3D(0, 0, 0):
                if world.parking_control is not None:
                    world.parking_control = None
                player_moving = False
                empty_space_relocation_finished = True
                if not camera_setting_for_line_detection:
                    world.camera_manager.set_camera_with_option(1, 5)
                    camera_setting_for_line_detection = True

            global Camera_image
            if empty_space_relocation_finished:
                lidar_result = world.get_result_parking_side()
                lane_detector = LaneDetector(Camera_image, lidar_result)
                lane_detector.run_lane_detection()

                world.move_to_line(lane_detector.get_move_distance())
                print('player location: ', world.player.get_location().y)
                print('location: ', abs(world.player.get_location().y - world.parking_relocation_position))

                if abs(world.player.get_location().y - world.parking_relocation_position) > 1.7 and move_possibility:
                    # print('player location: ', world.player.get_location().y)
                    # print('location: ', abs(world.player.get_location().y - world.parking_relocation_position))
                    world.player.apply_control(carla.VehicleControl(throttle=0.1, steer=0.0, brake=0.0, reverse=False))

                else:
                    move_possibility = None

                if move_possibility is None:
                    world.player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                    empty_space_relocation_finished = False
                    if lidar_result == 'l' or lidar_result == 'b':
                        world.parking_left = 1
                    elif lidar_result == 'r':
                        world.parking_right = 1

            print("parking side left", world.parking_left)
            print("parking side right", world.parking_right)

            if world.parking_left == 1:
                if abs(world.player.get_transform().rotation.yaw) < 60.5:
                    world.player.apply_control(
                        carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                    world.parking_left += 1
                    continue
                world.player.apply_control(carla.VehicleControl(throttle=0.2, steer=0.8, brake=0.0, reverse=False))
            elif world.parking_left == 2:
                if abs(world.player.get_transform().rotation.yaw) < 0.5:
                    world.player.apply_control(
                        carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                    world.parking_left += 1
                    continue
                world.player.apply_control(carla.VehicleControl(throttle=0.2, steer=-0.8, brake=0.0, reverse=True))
            elif world.parking_left == 3:
                if world.player.get_location().x < -6.5:
                    world.player.apply_control(
                        carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                    world.parking_left += 1
                    continue
                world.player.apply_control(carla.VehicleControl(throttle=0.2, steer=0, brake=0.0, reverse=True))

            if world.parking_right == 1:
                if abs(world.player.get_transform().rotation.yaw) > 120.5:
                    world.player.apply_control(
                        carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                    world.parking_right += 1
                    continue
                world.player.apply_control(carla.VehicleControl(throttle=0.2, steer=-0.8, brake=0.0, reverse=False))
            elif world.parking_right == 2:
                if abs(world.player.get_transform().rotation.yaw) > 178.0:
                    world.player.apply_control(
                        carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                    world.parking_right += 1
                    continue
                world.player.apply_control(carla.VehicleControl(throttle=0.2, steer=0.8, brake=0.0, reverse=True))
            elif world.parking_right == 3:
                if world.player.get_location().x > 3.5:
                    world.player.apply_control(
                        carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                    world.parking_right += 1
                    continue
                world.player.apply_control(carla.VehicleControl(throttle=0.2, steer=0, brake=0.0, reverse=True))


                # if not move_possibility:
                #     world.player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
                #
                # elif abs(world.player.get_location().y - world.parking_relocation_position) < 0.3 and move_possibility:
                #     print('location: ', abs(world.player.get_location().y - world.parking_relocation_position))
                #     world.player.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0, brake=0.0, reverse=False))
                #     move_possibility = False

            pygame.display.flip()

    finally:
        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            for sensor in world.camera_manager.sensor_list:
                sensor.destroy()
            world.destroy()

        pygame.quit()

        ##################### enter_the_parking_lot ########################
        # time.sleep(2)
        # if enter_the_parking_lot:
        #     if world.player.get_transform().rotation.yaw < 89.5:
        #         world.player.apply_control(
        #             carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=True))
        #         break
        #     if world.player.get_location().x < 19.0:
        #         world.player.apply_control(
        #             carla.VehicleControl(throttle=0.3, steer=-0.5, brake=0.0, reverse=False))
        #         continue
        #     world.player.apply_control(
        #         carla.VehicleControl(throttle=0.3, steer=0.5, brake=0.0))
        #
        ####################### find_a_parking_space ####################
        # if find_a_parking_space_first:
        #     if world.player.get_transform().rotation.yaw > 179.5:
        #         world.player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
        #         break
        #     if world.player.get_location().y > -20:
        #         world.player.apply_control(carla.VehicleControl(throttle=0.2, steer=0.32, brake=0.0, reverse=False))
        #         continue
        #     world.player.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0, brake=0.0, reverse=False))
        # if find_a_parking_space_second:
        #     if abs(world.player.get_transform().rotation.yaw) < 89.5:
        #         world.player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
        #         break
        #     world.player.apply_control(carla.VehicleControl(throttle=0.15, steer=0.3, brake=0.0, reverse=False))
        # if find_a_parking_space_third:
        #     if abs(world.player.get_transform().rotation.yaw) > 89.5:
        #         world.player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
        #         break
        #     if world.player.get_location().x > -1.5:
        #         world.player.apply_control(carla.VehicleControl(throttle=0.15, steer=-0.3, brake=0.0, reverse=False))
        #         continue
        #     world.player.apply_control(carla.VehicleControl(throttle=0.15, steer=0.15, brake=0.0, reverse=False))
        # if find_a_parking_space_fourth:
        #     if world.player.get_location().y < -39.5:
        #         world.player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
        #         break
        #     world.player.apply_control(carla.VehicleControl(throttle=0.1, steer=0.0, brake=0.0, reverse=False))
        #
        #
        ######################### park_on_the_right ##############################
        # if park_on_the_right_first:
        #     if abs(world.player.get_transform().rotation.yaw) > 120.5:
        #         world.player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
        #         break
        #     world.player.apply_control(carla.VehicleControl(throttle=0.2, steer=-0.8, brake=0.0, reverse=False))
        # if park_on_the_right_second:
        #     if abs(world.player.get_transform().rotation.yaw) > 178.0:
        #         world.player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
        #         break
        #     world.player.apply_control(carla.VehicleControl(throttle=0.2, steer=0.8, brake=0.0, reverse=True))
        # if park_on_the_right_third:
        #     if world.player.get_location().x > 3.5:
        #         world.player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
        #         break
        #     world.player.apply_control(carla.VehicleControl(throttle=0.2, steer=0, brake=0.0, reverse=True))
        # ######################### park_on_the_left ##############################
        # if park_on_the_left_first:
        #     if abs(world.player.get_transform().rotation.yaw) < 60.5:
        #         world.player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
        #         break
        #     world.player.apply_control(carla.VehicleControl(throttle=0.2, steer=0.8, brake=0.0, reverse=False))
        # if park_on_the_left_second:
        #     if abs(world.player.get_transform().rotation.yaw) < 0.5:
        #         world.player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
        #         break
        #     world.player.apply_control(carla.VehicleControl(throttle=0.2, steer=-0.8, brake=0.0, reverse=True))
        # if park_on_the_left_third:
        #     if world.player.get_location().x < -6.5:
        #         world.player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, reverse=False))
        #         break
        #     world.player.apply_control(carla.VehicleControl(throttle=0.2, steer=0, brake=0.0, reverse=True))


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.tesla.model3',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=50,
        type=int,
        help='number of walkers (default: 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enanble')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Random device seed')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enanble car lights')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    print(__doc__)

    try:

        client.load_world('Town05')
        client.reload_world()

        world = client.get_world()

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        if args.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False

        blueprints = world.get_blueprint_library().filter(args.filterv)
        blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # prepare the light state of the cars to spawn
            light_state = vls.NONE
            if args.car_lights_on:
                light_state = vls.Position | vls.LowBeam | vls.LowBeam

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                         .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                         .then(SetVehicleLightState(FutureActor, light_state)))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0  # how many pedestrians will run
        percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if not args.sync or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # example of how to use parameters
        traffic_manager.global_percentage_speed_difference(30.0)

        game_loop(args)

        while True:
            if args.sync and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

    finally:

        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
