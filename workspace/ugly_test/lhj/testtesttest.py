import carla
import random
import sys
import math
import pygame
from pygame.locals import *

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

# Get the world
world = client.get_world()

# Get the blueprint library
blueprint_library = world.get_blueprint_library()

# Define the spawn transform
spawn_transform = carla.Transform(carla.Location(x=0, y=-84.5, z=2), carla.Rotation())

# Define the target waypoint
target_location = carla.Location(x=24.3, y=-25, z=2)

# Spawn a vehicle
vehicle_bp = random.choice(blueprint_library.filter('vehicle'))
vehicle = world.spawn_actor(vehicle_bp, spawn_transform)

# Disable vehicle physics
vehicle.set_simulate_physics(False)


# Define a function to check if the vehicle reached the target waypoint
def check_reached_target(vehicle, target_location):
    vehicle_location = vehicle.get_location()
    distance_to_target = vehicle_location.distance(target_location)
    return distance_to_target < 1.0  # Define a threshold distance for reaching the target


# Define target velocity
target_velocity = 5  # m/s


# Define a function to calculate control commands
def calculate_control(vehicle, target_location, target_velocity):
    # Get current vehicle location and velocity
    vehicle_location = vehicle.get_location()
    vehicle_velocity = vehicle.get_velocity()

    # Calculate the direction vector towards the target location
    direction_vector = target_location - vehicle_location
    direction_vector_length = math.sqrt(direction_vector.x ** 2 + direction_vector.y ** 2 + direction_vector.z ** 2)
    normalized_direction_vector = carla.Vector3D(direction_vector.x / direction_vector_length,
                                                 direction_vector.y / direction_vector_length,
                                                 direction_vector.z / direction_vector_length)

    # Calculate throttle command
    throttle = min(target_velocity, direction_vector_length)

    # Calculate steering command (simple proportional control)
    # Note: This is a simple implementation and may need adjustments for better control
    desired_steering = math.atan2(normalized_direction_vector.y, normalized_direction_vector.x)
    current_yaw = math.atan2(vehicle_velocity.y, vehicle_velocity.x)
    steering = max(-1.0, min((desired_steering - current_yaw) / math.pi, 1.0))

    # Construct control command
    control = carla.VehicleControl()
    control.throttle = throttle
    control.steering = steering

    return control


# Define keyboard control
class KeyboardControl(object):
    def __init__(self, world):
        self._autopilot_enabled = False
        self.world = world
        self._control = carla.VehicleControl()

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                    self._autopilot_enabled = not self._autopilot_enabled
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    return True
        return False


try:
    pygame.init()
    display = pygame.display.set_mode((640, 480), pygame.HWSURFACE | pygame.DOUBLEBUF)
    hud = None
    controller = KeyboardControl(world)
    clock = pygame.time.Clock()

    while True:
        clock.tick_busy_loop(60)
        if controller.parse_events(world, clock):
            break

        # Calculate control command
        control = calculate_control(vehicle, target_location, target_velocity)

        # Get current vehicle location
        current_location = vehicle.get_location()

        # Get global route
        route = get_global_route(world.world_map, current_location, target_location)

        # Apply control command
        vehicle.apply_control(control)

        # Check if the vehicle reached the target waypoint
        if check_reached_target(vehicle, target_location):
            control.throttle = 0  # Stop the vehicle
            vehicle.apply_control(control)
            break  # Exit the loop if the vehicle reaches the target waypoint


finally:
    pygame.quit()
    # Destroy the vehicle
    vehicle.destroy()