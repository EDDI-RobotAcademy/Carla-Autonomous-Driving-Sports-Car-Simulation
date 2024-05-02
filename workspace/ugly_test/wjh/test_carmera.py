from __future__ import print_function

import glob
import os
import sys

try:
    sys.path.append(glob.glob(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# import carla
# import random
# import pygame
# from pygame.locals import *
import cv2
import numpy as np


def main():

    # 이미지를 읽어옵니다.1
    image = cv2.imread('parking_lot_image.jpg')

    # 이미지를 그레이스케일로 변환합니다.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러를 적용하여 노이즈를 제거합니다.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 엣지 검출을 수행합니다.
    edges = cv2.Canny(blur, 50, 150)

    # 직선을 감지합니다.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    # 검출된 직선을 원본 이미지에 그립니다.
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 결과를 표시합니다.
    cv2.imshow('Parking Lines Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
