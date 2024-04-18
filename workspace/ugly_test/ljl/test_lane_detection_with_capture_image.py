import unittest
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 선의 두 점 좌표를 계산
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0] # 선의 시작점의 y 좌표. 이미지 맨 아래
    y2 = int(y1*(3/5)) # 선의 끝 점의 y 좌표.
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


# 이미지에서 검출된 선들을 입력 받아 좌우 차선을 그림
def average_slope_intercept(image, lines):
    left_fit, right_fit = [], []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # 최적의 직선을 나타내는 기울기와 y 절편
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # print(parameters)
        slope = parameters[0] # 기울기
        intercept = parameters[1] # y절편
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
    # gray scaling
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # 이미지를 회색으로
    # blur treatment
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0) #블러
    # apply canny method
    canny_image = cv2.Canny(blurred_image, 50, 150)
    return canny_image


def region_of_interest(image):
    # specify the region we are interested in -> triangular
    height = image.shape[0]  # row value
    width = image.shape[1]  # column value
    # 다각형 생성. 아래의 다각형 = 삼각형.
    # 순서: 왼쪽 아래, 오른쪽 아래, 위
    # polygons = np.array([
    #     [(int(0.35*width), height), (int(0.7*width), height), (int(0.5*width), int(0.5*height))]
    # ])

    polygons = np.array([
        [(int(0.4 * width), height), (int(0.8 * width), height),
         (int(0.65 * width), int(0.75 * height)), (int(0.45 * width), int(0.75 * height))]
    ])

    # image 사이즈 만큼 0으로 차있는 배열 생성
    mask = np.zeros_like(image)
    # 이미지에 다각형을 그리고 해당 영역을 흰색으로 채움
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask) # 관심 영역만 표시됨.
    return masked_image


# 설정한 ROI가 어떻게 다각형으로 만들어 지는 지 보기 위함
def roi_image(image):
    height = image.shape[0]
    width = image.shape[1]

    # polygons = np.array([
    #     [(int(0.35 * width), height), (int(0.7 * width), height), (int(0.5 * width), int(0.5 * height))]
    # ])

    polygons = np.array([
        [(int(0.4 * width), height), (int(0.8 * width), height),
         (int(0.65 * width), int(0.75 * height)), (int(0.45 * width), int(0.75 * height))]
    ])

    roi_image = cv2.polylines(image, [polygons], True, (0, 0, 255), 10)

    return roi_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # print(line)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


class MyTestCase(unittest.TestCase):
    def test_line_recognition_image(self):
        image = cv2.imread('test04.png')
        lane_image = np.copy(image)
        canny_image = canny(lane_image)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(
            cropped_image, 1, np.pi / 180, 15, np.array([]), minLineLength=10, maxLineGap=20)
        averaged_lines = average_slope_intercept(lane_image, lines)
        line_image = display_lines(lane_image, averaged_lines)
        combined_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)


        # plt.imshow(canny_image)
        # plt.show()

        cv2.imshow('image', combined_image)
        cv2.imshow('polylines', roi_image(combined_image))
        cv2.imshow('edges', canny_image)
        cv2.waitKey(0)



if __name__ == '__main__':
    unittest.main()
