import unittest
import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    if not np.isnan(slope) and slope != 0:
        y1 = image.shape[0]
        y2 = 0
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    else:
        x1 = intercept
        x2 = intercept
        y1 = 0
        y2 = image.shape[1]

    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    line1_fit, line2_fit, line3_fit, line4_fit, line5_fit, line6_fit = [], [], [], [], [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:  # 기울기가 무한대이므로 NaN을 반환
            slope = np.nan
        else:
            slope = (y2 - y1) / (x2 - x1)

        intercept = y1 - slope * x1

        # slope = (y2-y1)/(x2-x1)
        # intercept = y1 - slope * x1

        if x1 < 700 and x2 < 700:
            if 500 < y1 < 600 and 500 < y2 < 600:
                line1_fit.append((slope, intercept))

            elif 200 < y1 < 400 and 200 < y2 < 400:
                line2_fit.append((slope, intercept))

            elif 0 < y1 < 200 and 0 < y2 < 200:
                line3_fit.append((slope, intercept))

        else:
            if 500 < y1 < 600 and 500 < y2 < 600:
                line4_fit.append((slope, intercept))

            elif 200 < y1 < 400 and 200 < y2 < 400:
                line5_fit.append((slope, intercept))

            elif 0 < y1 < 200 and 0 < y2 < 200:
                line6_fit.append((slope, intercept))

        plot_line(image, line[0])

    if line1_fit and line2_fit and line3_fit and line4_fit and line5_fit and line6_fit:
        line1_average = np.average(line1_fit, axis=0)
        line2_average = np.average(line2_fit, axis=0)
        line3_average = np.average(line3_fit, axis=0)
        line4_average = np.average(line4_fit, axis=0)
        line5_average = np.average(line5_fit, axis=0)
        line6_average = np.average(line6_fit, axis=0)

        line1 = make_coordinates(image, line1_average)
        line2 = make_coordinates(image, line2_average)
        line3 = make_coordinates(image, line3_average)
        line4 = make_coordinates(image, line4_average)
        line5 = make_coordinates(image, line5_average)
        line6 = make_coordinates(image, line6_average)

        print(line1_average)
        return np.array([line1, line2, line3, line4, line5, line6])


def average_points(image, lines):
    line1_fit, line2_fit, line3_fit, line4_fit, line5_fit, line6_fit = [], [], [], [], [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x1 < 700 and x2 < 700:
            if 500 < y1 < 600 and 500 < y2 < 600:
                line1_fit.append(np.array([x1, y1, x2, y2]))

            elif 200 < y1 < 400 and 200 < y2 < 400:
                line2_fit.append(np.array([x1, y1, x2, y2]))

            elif 0 < y1 < 200 and 0 < y2 < 200:
                line3_fit.append(np.array([x1, y1, x2, y2]))

        else:
            if 500 < y1 < 600 and 500 < y2 < 600:
                line4_fit.append(np.array([x1, y1, x2, y2]))

            elif 200 < y1 < 400 and 200 < y2 < 400:
                line5_fit.append(np.array([x1, y1, x2, y2]))

            elif 0 < y1 < 200 and 0 < y2 < 200:
                line6_fit.append(np.array([x1, y1, x2, y2]))

    if line1_fit and line2_fit and line3_fit and line4_fit and line5_fit and line6_fit:
        line1_average = np.average(line1_fit, axis=0)
        line2_average = np.average(line2_fit, axis=0)
        line3_average = np.average(line3_fit, axis=0)
        line4_average = np.average(line4_fit, axis=0)
        line5_average = np.average(line5_fit, axis=0)
        line6_average = np.average(line6_fit, axis=0)

        # print(line1_average)

        return np.array([line1_average, line2_average, line3_average, line4_average, line5_average, line6_average])


def min_max_points(image, lines):
    line1_fit, line2_fit, line3_fit, line4_fit, line5_fit, line6_fit = [], [], [], [], [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x1 < 700 and x2 < 700:
            if 500 < y1 < 600 and 500 < y2 < 600:
                line1_fit.append(np.array([x1, y1, x2, y2]))

            elif 200 < y1 < 400 and 200 < y2 < 400:
                line2_fit.append(np.array([x1, y1, x2, y2]))

            elif 0 < y1 < 200 and 0 < y2 < 200:
                line3_fit.append(np.array([x1, y1, x2, y2]))

        else:
            if 500 < y1 < 600 and 500 < y2 < 600:
                line4_fit.append(np.array([x1, y1, x2, y2]))

            elif 200 < y1 < 400 and 200 < y2 < 400:
                line5_fit.append(np.array([x1, y1, x2, y2]))

            elif 0 < y1 < 200 and 0 < y2 < 200:
                line6_fit.append(np.array([x1, y1, x2, y2]))

        plot_line(image, line[0])

    if line1_fit and line2_fit and line3_fit and line4_fit and line5_fit and line6_fit:
        line1_max = np.max(line1_fit, axis=0)
        line2_max = np.max(line2_fit, axis=0)
        line3_max = np.max(line3_fit, axis=0)
        line4_max = np.max(line4_fit, axis=0)
        line5_max = np.max(line5_fit, axis=0)
        line6_max = np.max(line6_fit, axis=0)

        line1_min = np.min(line1_fit, axis=0)
        line2_min = np.min(line2_fit, axis=0)
        line3_min = np.min(line3_fit, axis=0)
        line4_min = np.min(line4_fit, axis=0)
        line5_min = np.min(line5_fit, axis=0)
        line6_min = np.min(line6_fit, axis=0)

        line1 = np.array([line1_min[0], line1_min[1], line1_max[2], line1_min[3]])
        line2 = np.array([line2_min[0], line2_min[1], line2_max[2], line2_min[3]])
        line3 = np.array([line3_min[0], line3_min[1], line3_max[2], line3_min[3]])
        line4 = np.array([line4_min[0], line4_min[1], line4_max[2], line4_min[3]])
        line5 = np.array([line5_min[0], line5_min[1], line5_max[2], line5_min[3]])
        line6 = np.array([line6_min[0], line6_min[1], line6_max[2], line6_min[3]])

        return np.array([line1, line2, line3, line4, line5, line6])



def plot_line(image, line):
    x1, y1, x2, y2 = line
    plt.plot([x1, x2], [y1, y2], color='red')


def canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blurred_image, 50, 150)
    return canny_image


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]

    # test_roi
    polygons = np.array([
        [(int(0.09 * width), height), (width, height),
         (width, 0),  (int(0.09 * width), 0)]
    ])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def roi_image(image):
    height = image.shape[0]
    width = image.shape[1]

    # test_roi
    polygons = np.array([
        [(int(0.09 * width), height), (width, height),
         (width, 0), (int(0.09 * width), 0)]
    ])

    roi_image = cv2.polylines(image, [polygons], True, (0, 0, 255), 20)

    return roi_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)

    return line_image

def display_origin_line(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 10)

    return line_image



class MyTestCase(unittest.TestCase):
    def test_line_recognition_image(self):
        image = cv2.imread('test_parking_image/test04.png')
        lane_image = np.copy(image)
        canny_image = canny(lane_image)
        cropped_image = region_of_interest(canny_image)
        # minLineLength 검출할 선분의 최소 길이
        # maxLineGap 직선으로 간주할 엣지의 최대 간격
        lines = cv2.HoughLinesP(
            cropped_image, 1, np.pi / 180, 15, np.array([]), minLineLength=10, maxLineGap=10)


        min_max_lines = min_max_points(lane_image, lines)
        averaged_lines = average_points(lane_image, lines)
        line_image = display_lines(lane_image, min_max_lines)
        combined_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

        # 평균화(or 최대 최소 지점) 하기 전 원래의 line
        origin_line_image = display_origin_line(lane_image, lines)
        combined_origin_image = cv2.addWeighted(lane_image, 0.8, origin_line_image, 1, 1)

        # 그래프 상에서 위치 확인
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

        # scaled_combined_image = cv2.resize(combined_image, (0, 0), fx=0.5, fy=0.5)
        # scaled_lane_image = cv2.resize(lane_image, (0, 0), fx=0.5, fy=0.5)
        # scaled_canny_image = cv2.resize(canny_image, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow('image', combined_image)
        cv2.imshow('origin_lane_image', combined_origin_image)
        # cv2.imshow('polylines', roi_image(lane_image))
        cv2.imshow('edges', canny_image)
        cv2.waitKey(0)

if __name__ == '__main__':
    unittest.main()
