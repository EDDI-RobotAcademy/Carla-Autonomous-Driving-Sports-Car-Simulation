import carla  # the sim library itself
import time  # to set a delay after each photo
import cv2  # to work with images from cameras
import numpy as np  # in this example to change image representation - re-shaping
import math
import sys

sys.path.append('C:/CARLA_0.9.11/PythonAPI/carla')  # tweak to where you put carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

# connect to the sim
client = carla.Client('localhost', 2000)

# define speed contstants
PREFERRED_SPEED = 90  # what it says
SPEED_THRESHOLD = 2  # defines when we get close to desired speed so we drop the

# Max steering angle
MAX_STEER_DEGREES = 40

# camera mount offset on the car - you can tweak these to have the car in view or not
CAMERA_POS_Z = 3
CAMERA_POS_X = -5

# adding params to display text to image
font = cv2.FONT_HERSHEY_SIMPLEX
# org - defining lines to display telemetry values on the screen
org = (30, 30)  # this line will be used to show current speed
org2 = (30, 50)  # this line will be used for future steering angle
org3 = (30, 70)  # and another line for future telemetry outputs
org4 = (30, 90)  # and another line for future telemetry outputs
org3 = (30, 110)  # and another line for future telemetry outputs
fontScale = 0.5
# white color
color = (255, 255, 255)
# Line thickness of 2 px
thickness = 1


# maintain speed function
def maintain_speed(s):
    '''
    this is a very simple function to maintan desired speed
    s arg is actual current speed
    '''
    if s >= PREFERRED_SPEED:
        return 0
    elif s < PREFERRED_SPEED - SPEED_THRESHOLD:
        return 0.9  # think of it as % of "full gas"
    else:
        return 0.4  # tweak this if the car is way over or under preferred speed


# function to subtract 2 vectors
def angle_between(v1, v2):
    return math.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))


# function to get angle between the car and target waypoint
def get_angle(car, wp):
    '''
    this function to find direction to selected waypoint
    '''
    vehicle_pos = car.get_transform()
    car_x = vehicle_pos.location.x
    car_y = vehicle_pos.location.y
    wp_x = wp.transform.location.x
    wp_y = wp.transform.location.y

    # vector to waypoint
    x = (wp_x - car_x) / ((wp_y - car_y) ** 2 + (wp_x - car_x) ** 2) ** 0.5
    y = (wp_y - car_y) / ((wp_y - car_y) ** 2 + (wp_x - car_x) ** 2) ** 0.5

    # car vector
    car_vector = vehicle_pos.get_forward_vector()
    degrees = angle_between((x, y), (car_vector.x, car_vector.y))

    return degrees


world = client.get_world()
spawn_points = world.get_map().get_spawn_points()
# look for a blueprint of Mini car
vehicle_bp = world.get_blueprint_library().filter('*mini*')

start_point = spawn_points[0]
vehicle = world.try_spawn_actor(vehicle_bp[0], start_point)

# create and show the navigation route like in Tutorial 3
point_a = start_point.location  # we start at where the car is
sampling_resolution = 1
grp = GlobalRoutePlanner(world.get_map(), sampling_resolution)
# now let' pick the longest possible route
distance = 0
for loc in spawn_points:  # we start trying all spawn points
    # but we just exclude first at zero index
    cur_route = grp.trace_route(point_a, loc.location)
    if len(cur_route) > distance:
        distance = len(cur_route)
        route = cur_route
# draw the route in sim window - Note it does not get into the camera of the car
for waypoint in route:
    world.debug.draw_string(waypoint[0].transform.location, '^', draw_shadow=False,
                            color=carla.Color(r=0, g=0, b=255), life_time=30.0,
                            persistent_lines=True)

# setting RGB Camera - this follow the approach explained in a Carla video
# link: https://www.youtube.com/watch?v=om8klsBj4rc&t=1184s

camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')  # this ratio works in CARLA 9.14 on Windows
camera_bp.set_attribute('image_size_y', '360')

camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z, x=CAMERA_POS_X))
# this creates the camera in the sim
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)


def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))


image_w = camera_bp.get_attribute('image_size_x').as_int()
image_h = camera_bp.get_attribute('image_size_y').as_int()

camera_data = {'image': np.zeros((image_h, image_w, 4))}
# this actually opens a live stream from the camera
camera.listen(lambda image: camera_callback(image, camera_data))

cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
cv2.imshow('RGB Camera', camera_data['image'])

# main loop
quit = False
curr_wp = 5  # we will be tracking waypoints in the route and switch to next one wen we get close to current one
predicted_angle = 0
while curr_wp < len(route) - 1:
    # Carla Tick
    world.tick()
    if cv2.waitKey(1) == ord('q'):
        quit = True
        vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
        break
    image = camera_data['image']

    while curr_wp < len(route) and vehicle.get_transform().location.distance(route[curr_wp][0].transform.location) < 5:
        curr_wp += 1  # move to next wp if we are too close

    predicted_angle = get_angle(vehicle, route[curr_wp][0])
    image = cv2.putText(image, 'Steering angle: ' + str(round(predicted_angle, 3)), org, font, fontScale, color,
                        thickness, cv2.LINE_AA)
    v = vehicle.get_velocity()
    speed = round(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2), 0)
    image = cv2.putText(image, 'Speed: ' + str(int(speed)), org2, font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Next wp: ' + str(curr_wp), org3, font, fontScale, color, thickness, cv2.LINE_AA)
    estimated_throttle = maintain_speed(speed)
    # extra checks on predicted angle when values close to 360 degrees are returned
    if predicted_angle < -300:
        predicted_angle = predicted_angle + 360
    elif predicted_angle > 300:
        predicted_angle = predicted_angle - 360
    steer_input = predicted_angle
    # limit steering to max angel, say 40 degrees
    if predicted_angle < -MAX_STEER_DEGREES:
        steer_input = -MAX_STEER_DEGREES
    elif predicted_angle > MAX_STEER_DEGREES:
        steer_input = MAX_STEER_DEGREES
    # conversion from degrees to -1 to +1 input for apply control function
    steer_input = steer_input / 75

    vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))
    cv2.imshow('RGB Camera', image)

# clean up
cv2.destroyAllWindows()
camera.stop()
for sensor in world.get_actors().filter('*sensor*'):
    sensor.destroy()
for actor in world.get_actors().filter('*vehicle*'):
    actor.destroy()