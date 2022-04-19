from cgi import test
import cv2
import numpy as np
import matplotlib.pyplot as plt 
def hls_thresh(image, lower_bound, upper_bound):
    hls_img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hls_img, lower_bound, upper_bound)
    mask[mask == 255] = 1
    return mask

def hsv_thresh(image, lower_bound, upper_bound):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    mask[mask == 255] = 1
    return mask

def combine_thresh(image):

    # Thershold yellow
    yellow_lwr = (17, 45, 0)
    yellow_upr =  (32, 255, 255)
    yellow_threshold = hsv_thresh(image, yellow_lwr, yellow_upr)

    # HLS threshold

    hls_lwr = (7, 100, 100)
    hls_upr = (45, 255, 255)

    hls_threshold = hls_thresh(image, hls_lwr, hls_upr)

    combination = np.zeros_like(yellow_threshold)
    combination[ (hls_threshold == 1) | (yellow_threshold == 1) ] = 255

    
    return combination