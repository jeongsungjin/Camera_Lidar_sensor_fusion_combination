#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import tensorflow as tf
import os
from lanenet_model.lanenet_model import lanenet, lanenet_postprocess
from lanenet_model.local_utils.config_utils import parse_config_utils
from lanenet_model.local_utils.log_util import init_logger
from slidewindow_lanenet import SlideWindow_lanenet

# Config 파일 및 로그 설정
CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

# 모델 가중치 경로 설정
WEIGHTS_PATH = '/home/rsh/catkin_ws/src/lanenet_model/weight/tusimple_lanenet.ckpt'  # 실제 경로로 변경

# IPM Remap 파일 경로 설정
IPM_REMAP_FILE_PATH = '/home/rsh/catkin_ws/src/lanenet_model/data/tusimple_ipm_remap.yml'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # GPU 메모리를 필요에 따라 동적으로 할당
session = tf.compat.v1.Session(config=config)

# TensorFlow Eager Execution 비활성화
tf.compat.v1.disable_eager_execution()

class LaneNetProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.net = lanenet.LaneNet(phase='test', cfg=CFG)
        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG, ipm_remap_file_path=IPM_REMAP_FILE_PATH)

        self.input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='LaneNet')

        self.sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, WEIGHTS_PATH)
        LOG.info('Model weights loaded successfully')

        self.slidewindow = SlideWindow_lanenet()
        self.subscriber = rospy.Subscriber('/camera/image', Image, self.image_callback)

    def image_callback(self, data):
        try:
            # ROS 이미지 메시지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as e:
            # LOG.error(f'CvBridge Error: {e}')
            return

        # 프레임 리사이즈 및 전처리
        image_vis = cv_image
        image = cv2.resize(cv_image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0

        # 추론 실행
        try:
            binary_seg_image, instance_seg_image = self.sess.run(
                [self.binary_seg_ret, self.instance_seg_ret],
                feed_dict={self.input_tensor: [image]}
            )
        except Exception as e:
            # LOG.error(f'Error during inference: {e}')
            return

        # 후처리 결과
        postprocess_result = self.postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis,
            with_lane_fit=True,
            data_source='tusimple'
        )
        mask_image = postprocess_result['mask_image']

        # 이진 이미지를 원근 변환
        binary_image_for_warp = (binary_seg_image[0] * 255).astype(np.uint8)
        src_points = np.float32([
            [103, 250], [200, 182], [326, 182], [434, 250]
        ])
        dst_points = np.float32([
            [cv_image.shape[1] // 4, cv_image.shape[0]], [cv_image.shape[1] // 4, 0],
            [cv_image.shape[1] * 3 // 4, 0], [cv_image.shape[1] * 3 // 4, cv_image.shape[0]]
        ])
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        binary_image_warped = cv2.warpPerspective(binary_image_for_warp, matrix, (512, 256))

        # 슬라이딩 윈도우 적용
        lane_line_markings, x_location, another_result = self.slidewindow.slidewindow(binary_image_warped, roi_flag=0)

        # 결과 출력
        cv2.imshow('Mask Image', cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))
        cv2.imshow('Source Image', cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
        cv2.imshow('Binary Image', binary_image_for_warp)
        cv2.imshow('Warped Binary Image', binary_image_warped)
        cv2.imshow('Lane Line Markings', cv2.cvtColor(lane_line_markings, cv2.COLOR_RGB2BGR))
        print("x_location", x_location)
        cv2.waitKey(1)

def main():
    rospy.init_node('lanenet_processor', anonymous=True)
    LaneNetProcessor()
    rospy.spin()

if __name__ == '__main__':
    main()
