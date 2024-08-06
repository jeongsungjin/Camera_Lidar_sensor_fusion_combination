#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))
from slidewindow import SlideWindow

class LaneDetectionROS:
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('lane_detection_ros', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)

        # SlideWindow 객체 초기화
        self.slidewindow = SlideWindow()

        # 트랙바 초기값 설정
        self.lower_yellow = np.array([20, 110, 40])
        self.upper_yellow = np.array([53, 240, 255])
        self.lower_white = np.array([10, 0, 214])
        self.upper_white = np.array([82, 140, 255])

        # 트랙바 창 생성
        cv2.namedWindow("Trackbars")
        self.create_trackbars()

        # 트랙바와 이미지 윈도우 창을 띄운 상태로 ROS 스핀
        rospy.spin()

    def image_callback(self, msg):
        try:
            # ROS 이미지 메시지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 비디오 프레임 크기 조정
            frame_resized = cv2.resize(cv_image, (640, 480))

            # 이미지 처리
            y, x = frame_resized.shape[0:2]

            # HSV 변환 및 마스크 생성
            img_hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

            h, s, v = cv2.split(img_hsv)

            # 트랙바로부터 HSV 값 읽기
            self.read_trackbar_values()

            # 노란색 및 흰색 마스크 생성
            mask_yellow = cv2.inRange(img_hsv, self.lower_yellow, self.upper_yellow)
            mask_white = cv2.inRange(img_hsv, self.lower_white, self.upper_white)
            filtered_yellow = cv2.bitwise_and(frame_resized, frame_resized, mask=mask_yellow)
            filtered_white = cv2.bitwise_and(frame_resized, frame_resized, mask=mask_white)
            masks = cv2.bitwise_or(mask_yellow, mask_white)
            filtered_img = cv2.bitwise_and(frame_resized, frame_resized, mask=masks)

            # Perspective Transform
            left_margin = 250
            top_margin = 320
            src_point1 = [100, 460]      # 왼쪽 아래
            src_point2 = [left_margin+20, top_margin]
            src_point3 = [x-left_margin-20, top_margin]
            src_point4 = [x -100, 460]  

            src_points = np.float32([src_point1, src_point2, src_point3, src_point4])

            dst_point1 = [x//4, 460]    # 왼쪽 아래
            dst_point2 = [x//4, 0]      # 왼쪽 위
            dst_point3 = [x//4*3, 0]    # 오른쪽 위
            dst_point4 = [x//4*3, 460]  # 오른쪽 아래

            dst_points = np.float32([dst_point1, dst_point2, dst_point3, dst_point4])

            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            warped_img = cv2.warpPerspective(filtered_img, matrix, (640, 480))

            # 이미지 평행이동
            translated_img = self.translate_image(warped_img, tx=0, ty=0)

            # 기존 HSV 방식에서 다시 살리기
            grayed_img = cv2.cvtColor(translated_img, cv2.COLOR_BGR2GRAY)

            # 이미지 이진화
            bin_img = np.zeros_like(grayed_img)
            bin_img[grayed_img > 20] = 1

            # 슬라이딩 윈도우 차선 검출
            out_img, x_location, _ = self.slidewindow.slidewindow(bin_img, False)

            # 결과 표시
            # cv2.imshow('Original Image', frame_resized)
            # cv2.imshow("Yellow Mask", filtered_yellow)
            # cv2.imshow("White Mask", filtered_white)
            # cv2.imshow("Filtered Image", filtered_img)
            # cv2.imshow("Warped Image", warped_img)
            # cv2.imshow("Output Image", out_img)
            
            # 화면 업데이트 및 이벤트 처리
            # cv2.waitKey(1)  # 1ms 동안 대기

            # 이미지 저장
            cv2.imwrite('/tmp/original.png', frame_resized)
            cv2.imwrite('/tmp/output.png', out_img)
            print("x_location", x_location)

        except Exception as e:
            rospy.logerr("Error processing image: %s" % str(e))

    def translate_image(self, image, tx, ty):
        rows, cols = image.shape[:2]

        # 평행이동 행렬 생성
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

        # 이미지 평행이동
        translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))

        return translated_image

    def create_trackbars(self):
        # 노란색 HSV 범위 트랙바
        cv2.createTrackbar("Yellow Lower H", "Trackbars", self.lower_yellow[0], 180, self.nothing)
        cv2.createTrackbar("Yellow Lower S", "Trackbars", self.lower_yellow[1], 255, self.nothing)
        cv2.createTrackbar("Yellow Lower V", "Trackbars", self.lower_yellow[2], 255, self.nothing)
        cv2.createTrackbar("Yellow Upper H", "Trackbars", self.upper_yellow[0], 180, self.nothing)
        cv2.createTrackbar("Yellow Upper S", "Trackbars", self.upper_yellow[1], 255, self.nothing)
        cv2.createTrackbar("Yellow Upper V", "Trackbars", self.upper_yellow[2], 255, self.nothing)

        # 흰색 HSV 범위 트랙바
        cv2.createTrackbar("White Lower H", "Trackbars", self.lower_white[0], 180, self.nothing)
        cv2.createTrackbar("White Lower S", "Trackbars", self.lower_white[1], 255, self.nothing)
        cv2.createTrackbar("White Lower V", "Trackbars", self.lower_white[2], 255, self.nothing)
        cv2.createTrackbar("White Upper H", "Trackbars", self.upper_white[0], 180, self.nothing)
        cv2.createTrackbar("White Upper S", "Trackbars", self.upper_white[1], 255, self.nothing)
        cv2.createTrackbar("White Upper V", "Trackbars", self.upper_white[2], 255, self.nothing)

    def read_trackbar_values(self):
        self.lower_yellow[0] = cv2.getTrackbarPos("Yellow Lower H", "Trackbars")
        self.lower_yellow[1] = cv2.getTrackbarPos("Yellow Lower S", "Trackbars")
        self.lower_yellow[2] = cv2.getTrackbarPos("Yellow Lower V", "Trackbars")
        self.upper_yellow[0] = cv2.getTrackbarPos("Yellow Upper H", "Trackbars")
        self.upper_yellow[1] = cv2.getTrackbarPos("Yellow Upper S", "Trackbars")
        self.upper_yellow[2] = cv2.getTrackbarPos("Yellow Upper V", "Trackbars")

        self.lower_white[0] = cv2.getTrackbarPos("White Lower H", "Trackbars")
        self.lower_white[1] = cv2.getTrackbarPos("White Lower S", "Trackbars")
        self.lower_white[2] = cv2.getTrackbarPos("White Lower V", "Trackbars")
        self.upper_white[0] = cv2.getTrackbarPos("White Upper H", "Trackbars")
        self.upper_white[1] = cv2.getTrackbarPos("White Upper S", "Trackbars")
        self.upper_white[2] = cv2.getTrackbarPos("White Upper V", "Trackbars")

    def nothing(self, x):
        pass

if __name__ == "__main__":
    LaneDetectionROS()
