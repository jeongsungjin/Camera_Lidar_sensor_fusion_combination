
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
# from slidewindow import SlideWindow  # Import SlideWindow

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

tf.compat.v1.disable_eager_execution()

TOTAL_CNT = 50

class SlideWindow:
    def __init__(self):
        self.current_line = "DEFAULT"
        self.left_fit = None
        self.right_fit = None
        self.leftx = None
        self.rightx = None
        self.lhd = 240
        self.left_cnt = 25
        self.right_cnt = 25
        self.x_previous = 256

    def slidewindow(self, img, roi_flag):
        height = img.shape[0]
        width = img.shape[1]

        # Initialize output image (for visualization)
        out_img = np.dstack((img, img, img)) * 255

        # Parameters for sliding windows
        window_height = 15  # Adjusted height of each window
        nwindows = 20  # Number of windows

        # Find nonzero locations in img
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 40
        minpix = 50

        # Initialize variables
        left_lane_inds = []
        right_lane_inds = []

        # Define window boundaries
        win_l_w_l = 100
        win_l_w_r = 200
        win_r_w_l = width - 200
        win_r_w_r = width - 100
        circle_height = 100

        road_width = 0.455
        half_road_width = 0.5 * road_width

        # Draw initial sliding window boundaries
        pts_left = np.array([[win_l_w_l, height], [win_l_w_l, height - window_height], [win_l_w_r, height - window_height], [win_l_w_r, height]], np.int32)
        cv2.polylines(out_img, [pts_left], True, (0,255,0), 1)
        
        pts_right = np.array([[win_r_w_l, height], [win_r_w_l, height - window_height], [win_r_w_r, height - window_height], [win_r_w_r, height]], np.int32)
        cv2.polylines(out_img, [pts_right], True, (255,0,0), 1)

        pts_catch = np.array([[0, circle_height], [width, circle_height]], np.int32)
        cv2.polylines(out_img, [pts_catch], False, (0,120,120), 1)

        # Find good indices for left and right lane lines
        good_left_inds = ((nonzerox >= win_l_w_l) & (nonzeroy < height) & (nonzeroy >= height - window_height) & (nonzerox <= win_l_w_r)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_r_w_l) & (nonzeroy < height) & (nonzeroy >= height - window_height) & (nonzerox <= win_r_w_r)).nonzero()[0]

        # Initial x position
        x_current = None
        line_flag = 0
        x_location = self.x_previous  # Initialize x_location

        if len(good_left_inds) > len(good_right_inds):
            line_flag = 1
            x_current = int(np.mean(nonzerox[good_left_inds]))

        elif len(good_left_inds) < len(good_right_inds):
            line_flag = 2
            x_current = int(np.mean(nonzerox[good_right_inds]))

        else:
            line_flag = 3

        # Visualization of the good indices
        if line_flag == 1:
            for i in range(len(good_left_inds)):
                out_img = cv2.circle(out_img, (nonzerox[good_left_inds[i]], nonzeroy[good_left_inds[i]]), 1, (0,255,0), -1)
        elif line_flag == 2:
            for i in range(len(good_right_inds)):
                out_img = cv2.circle(out_img, (nonzerox[good_right_inds[i]], nonzeroy[good_right_inds[i]]), 1, (255,0,0), -1)

        # Sliding window processing
        for window in range(nwindows):
            if line_flag == 1:
                win_y_low = height - (window + 1) * window_height
                win_y_high = height - window * window_height
                win_x_low = x_current - margin
                win_x_high = x_current + margin

                cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 1)
                cv2.rectangle(out_img, (win_x_low + int(width * road_width), win_y_low), (win_x_high + int(width * road_width), win_y_high), (255, 0, 0), 1)

                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

                if len(good_left_inds) > minpix:
                    x_current = int(np.mean(nonzerox[good_left_inds]))
                elif len(left_lane_inds) > 0:
                    p_left = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2) 
                    x_current = int(np.polyval(p_left, win_y_high))

                if circle_height - 10 <= win_y_low < circle_height + 10:
                    x_location = int(x_current + width * half_road_width)
                    cv2.circle(out_img, (x_location, circle_height), 10, (0, 0, 255), 5)

            elif line_flag == 2:
                win_y_low = height - (window + 1) * window_height
                win_y_high = height - window * window_height
                win_x_low = x_current - margin
                win_x_high = x_current + margin

                cv2.rectangle(out_img, (win_x_low - int(width * road_width), win_y_low), (win_x_high - int(width * road_width), win_y_high), (0, 255, 0), 1)
                cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (255, 0, 0), 1)

                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

                if len(good_right_inds) > minpix:
                    x_current = int(np.mean(nonzerox[good_right_inds]))
                elif len(right_lane_inds) > 0:
                    p_right = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2) 
                    x_current = int(np.polyval(p_right, win_y_high))

                if circle_height - 10 <= win_y_low < circle_height + 10:
                    x_location = int(x_current - width * half_road_width)
                    cv2.circle(out_img, (x_location, circle_height), 10, (0, 0, 255), 5)
            
            else: # Can't see lanes
                x_location = self.x_previous
                cv2.circle(out_img, (x_location, circle_height), 10, (0, 0, 255), 5)

            self.x_previous = x_location

        return out_img, x_location, self.current_line
    
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
    assert ops.exists(video_path), '{} not exist'.format(video_path)

    LOG.info('Start processing video')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        LOG.error('Error opening video file')
        return

    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # Set sess configuration
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.compat.v1.Session(config=sess_config)

    # Define moving average version of the learned variables for eval
    with tf.compat.v1.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # Define saver
    saver = tf.compat.v1.train.Saver(variables_to_restore)

    with sess.as_default():
        try:
            saver.restore(sess=sess, save_path=weights_path)
            LOG.info('Model weights loaded successfully')
        except Exception as e:
            LOG.error('Error loading model weights: {}'.format(e))
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                LOG.info('No more frames to process or error reading frame')
                break

            # Resize and preprocess frame
            image_vis = frame
            image = cv2.resize(frame, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0

            # Run inference
            try:
                binary_seg_image, instance_seg_image = sess.run(
                    [binary_seg_ret, instance_seg_ret],
                    feed_dict={input_tensor: [image]}
                )
            except Exception as e:
                LOG.error('Error during inference: {}'.format(e))
                continue

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
                LOG.info('Model has fitted {} lanes'.format(len(lane_params)))
                for i in range(len(lane_params)):
                    LOG.info('Fitted 2-order lane {} curve param: {}'.format(i + 1, lane_params[i]))

            for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
                instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
            embedding_image = np.array(instance_seg_image[0], np.uint8)

            frame_resized = cv2.resize(frame, (512, 256))
            y, x = frame_resized.shape[0:2]
            # 소스 포인트 설정
            left_margin = 200
            top_margin = 180

            # src_points = np.float32([
            #     [80, y - 20],        # 왼쪽 아래
            #     [left_margin, top_margin], # 왼쪽 위
            #     [x - left_margin, top_margin], # 오른쪽 위
            #     [x - 80, y - 20]     # 오른쪽 아래
            # ])

            src_points = np.float32([
                [103, 250],        # 왼쪽 아래
                [200, 182], # 왼쪽 위
                [326, 182], # 오른쪽 위
                [434, 250]     # 오른쪽 아래
            ])

            # 목적지 포인트 설정 (직사각형의 형태를 목표로 함)
            dst_points = np.float32([
                [x // 4, y],             # 왼쪽 아래
                [x // 4, 0],             # 왼쪽 위
                [x * 3 // 4, 0],         # 오른쪽 위
                [x * 3 // 4, y]          # 오른쪽 아래
            ])

            # Check source and destination points
            LOG.info('Source Points: {}'.format(src_points))
            LOG.info('Destination Points: {}'.format(dst_points))

            # Calculate perspective transform matrix
            try:
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                LOG.info('Perspective transform matrix: {}'.format(matrix))
            except Exception as e:
                LOG.error('Error in calculating perspective transform matrix: {}'.format(e))
                continue

            # Check the matrix and input image
            LOG.info('Binary Segmentation Image Shape: {}'.format(binary_seg_image[0].shape))
            LOG.info('Binary Segmentation Image Min/Max Values: Min: {}, Max: {}'.format(
                np.min(binary_seg_image[0]), np.max(binary_seg_image[0])
            ))

            # Ensure binary image is uint8
            binary_image_for_warp = (binary_seg_image[0] * 255).astype(np.uint8)
            LOG.info('Binary Image for Warp Shape: {}'.format(binary_image_for_warp.shape))
            LOG.info('Binary Image for Warp Min/Max Values: Min: {}, Max: {}'.format(
                np.min(binary_image_for_warp), np.max(binary_image_for_warp)
            ))

            # Warp the binary image using the perspective transform matrix
            try:
                binary_image_warped = cv2.warpPerspective(binary_image_for_warp, matrix, (512, 256))
                LOG.info('Warped Binary Image Shape: {}'.format(binary_image_warped.shape))
            except Exception as e:
                LOG.error('OpenCV Error during warpPerspective: {}'.format(e))
                continue

            # Apply sliding window
            slidewindow = SlideWindow()
            lane_line_markings, x_location, another_result = slidewindow.slidewindow(binary_image_warped, roi_flag=0)
            # Show results
            cv2.imshow('Mask Image', cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))
            cv2.imshow('Source Image', cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
            cv2.imshow('Instance Image', cv2.cvtColor(embedding_image, cv2.COLOR_RGB2BGR))
            cv2.imshow('Binary Image', (binary_seg_image[0] * 255).astype(np.uint8))
            cv2.imshow('Warped Binary Image', binary_image_warped)
            cv2.imshow('Lane Line Markings', cv2.cvtColor(lane_line_markings, cv2.COLOR_RGB2BGR))
            print("차선 중심 x좌표 : ", x_location)
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        sess.close()
        cv2.destroyAllWindows()

        LOG.info('Video processing complete.')

if __name__ == '__main__':
    # Initialize args
    args = init_args()
    test_lanenet(args.video_path, args.weights_path, with_lane_fit=args.with_lane_fit)
