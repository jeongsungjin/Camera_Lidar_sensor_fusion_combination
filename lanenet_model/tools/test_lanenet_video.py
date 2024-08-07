#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on video file
"""
import argparse
import os.path as ops
import time

import cv2
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
    Initialize command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='The path to the input video file')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--with_lane_fit', type=args_str2bool, help='If need to do lane fit', default=True)

    return parser.parse_args()


def args_str2bool(arg_value):
    """
    Convert string to boolean.
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """
    Normalize the input array to the range [0, 255].
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(video_path, weights_path, with_lane_fit=True):
    """
    Process video frames with LaneNet model and display results.
    """
    assert ops.exists(video_path), '{:s} not exist'.format(video_path)

    LOG.info('Start processing video')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        LOG.error('Error opening video file')
        return

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # Define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # Define saver
    saver = tf.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and preprocess frame
            image_vis = frame
            image = cv2.resize(frame, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0

            # Run inference
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )

            # Postprocess results
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
                LOG.info('Model have fitted {:d} lanes'.format(len(lane_params)))
                for i in range(len(lane_params)):
                    LOG.info('Fitted 2-order lane {:d} curve param: {}'.format(i + 1, lane_params[i]))

            for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
                instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
            embedding_image = np.array(instance_seg_image[0], np.uint8)

            
            # Show results
            cv2.imshow('Mask Image', cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))
            cv2.imshow('Source Image', cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
            cv2.imshow('Instance Image', cv2.cvtColor(embedding_image, cv2.COLOR_RGB2BGR))
            cv2.imshow('Binary Image', (binary_seg_image[0] * 255).astype(np.uint8))

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    sess.close()
    cv2.destroyAllWindows()

    LOG.info('Video processing complete.')


if __name__ == '__main__':
    """
    Test code
    """
    # Initialize args
    args = init_args()

    test_lanenet(args.video_path, args.weights_path, with_lane_fit=args.with_lane_fit)
