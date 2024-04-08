import unittest
import cv2


class MyTestCase(unittest.TestCase):
    def test_something(self):
        print('test_image_recognition')
        image = cv2.imread('test_image.jpg')
        cv2.imshow('image', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    unittest.main()
