import carla
import random

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

# Get the world
world = client.get_world()

# Get the blueprint library
blueprint_library = world.get_blueprint_library()

# Define the spawn transform
spawn_transform = carla.Transform(carla.Location(x=0, y=-84.5, z=2), carla.Rotation())

# Spawn a vehicle
vehicle_bp = random.choice(blueprint_library.filter('vehicle'))
vehicle = world.spawn_actor(vehicle_bp, spawn_transform)

# Disable vehicle physics
vehicle.set_simulate_physics(False)

# Define the target location
target_location = carla.Location(x=24.3, y=-25, z=2)

try:
    # Find the waypoint nearest to the target location
    target_waypoint = world.get_map().get_waypoint(target_location)

    # Move the vehicle to the target waypoint
    vehicle.set_transform(target_waypoint.transform)

finally:
    # Destroy the vehicle
    vehicle.destroy()