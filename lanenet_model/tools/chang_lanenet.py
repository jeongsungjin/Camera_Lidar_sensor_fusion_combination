import argparse
import os.path as ops
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

tf.compat.v1.disable_eager_execution()

def init_args():
    """
    명령행 인자 초기화
    :return: 파싱된 명령행 인자
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='이미지 경로 또는 src 이미지 저장 디렉토리')
    parser.add_argument('--weights_path', type=str, help='모델 가중치 경로')
    parser.add_argument('--with_lane_fit', type=args_str2bool, help='차선 맞춤이 필요한지 여부', default=True)

    return parser.parse_args()

def args_str2bool(arg_value):
    """
    명령행 인자 문자열을 boolean으로 변환
    :param arg_value: 명령행 인자 문자열
    :return: Boolean 값
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('지원되지 않는 값입니다.')

def minmax_scale(input_arr):
    """
    입력 배열을 0-255 범위로 스케일링
    :param input_arr: 입력 배열
    :return: 스케일링된 배열
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

def perspective_transform(image):
    """
    입력 이미지에 원근 변환을 적용하여 차선을 일직선으로 보정
    :param image: 입력 이미지
    :return: 변환된 이미지
    """
    height, width = image.shape[:2]

    # 소스 포인트 정의 (이미지 내 차선 마킹 기준으로 수동 선택)
    src = np.float32([[width // 2 - 30, height * 0.3],
                      [width // 2 + 30, height * 0.3],
                      [width * 0.1, 0],
                      [width * 0.9, 0]])

    # 소스 포인트를 매핑할 목적지 포인트 정의
    dst = np.float32([[width * 0.25, 0],
                      [width * 0.75, 0],
                      [width * 0.25, height],
                      [width * 0.75, height]])

    # 원근 변환 행렬 계산
    M = cv2.getPerspectiveTransform(src, dst)

    # 원근 변환 행렬을 사용하여 이미지 변환
    warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)

    return warped

def test_lanenet(image_path, weights_path, with_lane_fit=True):
    """
    LaneNet 모델 테스트를 위한 차선 검출
    :param image_path: 입력 이미지 경로
    :param weights_path: 모델 가중치 경로
    :param with_lane_fit: 차선 맞춤이 필요한지 여부
    :return: 없음
    """
    assert ops.exists(image_path), '{:s} 존재하지 않음'.format(image_path)

    LOG.info('이미지 읽기 및 전처리 시작')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image.copy()
    
    # 차선을 일직선으로 보정하기 위해 원근 변환 적용
    image = perspective_transform(image)
    
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0
    LOG.info('이미지 로드 완료, 시간 소요: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # 세션 설정
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.compat.v1.Session(config=sess_config)

    # 평가를 위한 학습된 변수의 이동 평균 버전 정의
    with tf.compat.v1.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # saver 정의
    saver = tf.compat.v1.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        loop_times = 500
        for i in range(loop_times):
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
        t_cost = time.time() - t_start
        t_cost /= loop_times
        LOG.info('단일 이미지 추론 시간 소요: {:.5f}s'.format(t_cost))

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis,
            with_lane_fit=with_lane_fit,
            data_source='tusimple'
        )
        mask_image = postprocess_result['mask_image']
        if with_lane_fit:
            lane_params = postprocess_result['fit_params']
            LOG.info('모델이 {:d} 차선을 맞춤'.format(len(lane_params)))
            for i in range(len(lane_params)):
                LOG.info('2차 차선 {:d} 곡선 맞춤 파라미터: {}'.format(i + 1, lane_params[i]))

        for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        plt.figure('mask_image')
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        plt.show()

    sess.close()

    return

if __name__ == '__main__':
    """
    테스트 코드
    """
    # 명령행 인자 초기화
    args = init_args()

    test_lanenet(args.image_path, args.weights_path, with_lane_fit=args.with_lane_fit)