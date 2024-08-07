import cv2
import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import *
from matplotlib.pyplot import *
import math
# float로 조향값 public

TOTAL_CNT = 50

class SlideWindow_lanenet:
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