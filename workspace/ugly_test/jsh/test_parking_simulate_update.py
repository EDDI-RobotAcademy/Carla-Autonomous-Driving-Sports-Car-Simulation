import time
import random
import carla
import pygame

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

class TestParkingSimulateUpdate():

    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()

        self.world = carla.Client('localhost', 2000).get_world()
        self.actor_list = []
        blueprint_library = self.world.get_blueprint_library()

        bp = random.choice(blueprint_library.filter('vehicle.tesla.cybertruck'))
        init_pos = carla.Transform(carla.Location(x=-43.9, y=-16.9, z=0.5), carla.Rotation(yaw=270))
        self.vehicle = self.world.spawn_actor(bp, init_pos)
        self.actor_list.append(self.vehicle)

        self.parking_points = [
            carla.Location(x=14, y=-42),
            carla.Location(x=14, y=-13),
            carla.Location(x=-33, y=-13),
            carla.Location(x=-33, y=-42)
        ]

        self.start_points = []
        self.current_point_index = None

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
        self.set_start_point()

    def set_start_point(self):
        vehicle_location = self.vehicle.get_location()
        min_distance = float('inf')
        min_index = None
        for idx, point in enumerate(self.parking_points):
            distance = np.linalg.norm(np.array([vehicle_location.x, vehicle_location.y]) - np.array([point.x, point.y]))
            if distance < min_distance:
                min_distance = distance
                min_index = idx
        self.current_point_index = min_index

    def moving_parking(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    running = False

            vehicle_location = self.vehicle.get_location()

            min_distance = float('inf')
            min_index = None
            for idx, point in enumerate(self.parking_points):
                distance = np.linalg.norm(
                    np.array([vehicle_location.x, vehicle_location.y]) - np.array([point.x, point.y]))
                if distance < min_distance:
                    min_distance = distance
                    min_index = idx

            next_index = (min_index + 1) % len(self.parking_points)
            next_point = self.parking_points[next_index]

            distance = np.linalg.norm(
                np.array([next_point.x, next_point.y]) - np.array([vehicle_location.x, vehicle_location.y]))

            if distance < 20.0:
                self.current_point_index = next_index

            else:
                target_rotation = np.degrees(
                    np.arctan2(next_point.y - vehicle_location.y, next_point.x - vehicle_location.x))
                self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=(target_rotation - self.vehicle.get_transform().rotation.yaw) / 180.0,
                                                                brake=0.0))

            self.clock.tick(60)

        return

    def destroy(self):
        print('destroying actors')
        for actor in self.actor_list:
            actor.destroy()
        print('done.')

    def run(self):
        self.move_to_init_parking()
        self.moving_parking()
        pygame.quit()


def main():
    ego_vehicle = TestParkingSimulateUpdate()
    try:
        ego_vehicle.run()
    finally:
        if ego_vehicle is not None:
            ego_vehicle.destroy()

if __name__ == '__main__':
    main()