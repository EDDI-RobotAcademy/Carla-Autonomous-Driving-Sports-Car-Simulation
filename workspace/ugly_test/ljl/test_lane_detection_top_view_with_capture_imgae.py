import unittest
import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit, right_fit = [], []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    # print(left_fit)
    # print(right_fit)
    if left_fit and right_fit:
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])
    else:
        return None


def canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blurred_image, 50, 150)
    return canny_image


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]

    # test_image
    polygons = np.array([
        [(int(0.2 * width + 150), height), (int(0.7 * width + 150), height),
         (int(0.57 * width + 150), int(0.8 * height)), (int(0.33 * width + 150), int(0.8 * height))]
    ])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def roi_image(image):
    height = image.shape[0]
    width = image.shape[1]

    # test_image
    polygons = np.array([
        [(int(0.2 * width + 150), height), (int(0.7 * width + 150), height),
         (int(0.57 * width + 150), int(0.8 * height)), (int(0.33 * width + 150), int(0.8 * height))]
    ])

    roi_image = cv2.polylines(image, [polygons], True, (0, 0, 255), 10)

    return roi_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_image


class MyTestCase(unittest.TestCase):
    def test_line_recognition_image(self):
        image = cv2.imread('test_parking_image/test_parking_image09.png')
        lane_image = np.copy(image)
        canny_image = canny(lane_image)
        # cropped_image = region_of_interest(canny_image)
        # minLineLength 검출할 선분의 최소 길이
        # maxLineGap 직선으로 간주할 엣지의 최대 간격
        lines = cv2.HoughLinesP(
            canny_image, 1, np.pi / 180, 50, np.array([]), minLineLength=50, maxLineGap=20)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(lane_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # averaged_lines = average_slope_intercept(lane_image, lines)
        # line_image = display_lines(lane_image, averaged_lines)
        combined_image = cv2.addWeighted(image, 0.8, lane_image, 1, 1)
        # combined_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)


        # cv2.imshow('image', combined_image)
        cv2.imshow('lane image', lane_image)
        # cv2.imshow('polylines', roi_image(combined_image))
        cv2.imshow('edges', canny_image)
        cv2.waitKey(0)

if __name__ == '__main__':
    unittest.main()
