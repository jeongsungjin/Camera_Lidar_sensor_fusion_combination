#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def publish_image():
    rospy.init_node('image_publisher', anonymous=True)
    image_pub = rospy.Publisher('camera/image', Image, queue_size=10)
    bridge = CvBridge()

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture('/dev/video4')

    if not cap.isOpened():
        rospy.logerr("카메라를 열 수 없습니다.")
        return

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logerr("프레임을 읽을 수 없습니다.")
            break

        # OpenCV 이미지를 ROS 이미지 메시지로 변환
        image_msg = bridge.cv2_to_imgmsg(frame, "bgr8")

        # 퍼블리시
        image_pub.publish(image_msg)

        # 30 FPS로 퍼블리시 (필요에 따라 조정 가능)
        rospy.Rate(30).sleep()

    cap.release()

if __name__ == '__main__':
    try:
        publish_image()
    except rospy.ROSInterruptException:
        pass
