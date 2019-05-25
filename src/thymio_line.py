import numpy as np
import cv2, sys
from matplotlib import pyplot as plt
import rospy, rosnode, rostopic, rosmsg, std_msgs.msg, sensor_msgs.msg, nav_msgs.msg
from sensor_msgs.msg import Range, Image, CompressedImage
from PID import PID
from tracker import tracker

class Thymio:
    
    def __init__(self, thymio_name):
        """init"""
        self.thymio_name = thymio_name

        print(self.thymio_name)

        self.frame = []

        rospy.init_node('hand_following_thymio_controller', anonymous=True)

        self.angular_pid = PID(Kd=5, Ki=0, Kp=0.5)
        self.linear_pid = PID(Kd=5, Ki=0, Kp=0.5)
        self.object_pid = PID(Kd=3, Ki=0, Kp=0.5)

        self.total_rectangle = 9
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.traverse_point = []

        self.camera_subscriber = rospy.Subscriber(self.thymio_name + '/camera/image_raw/compressed', CompressedImage, self.camera_callback_compressed, queue_size=1, buff_size=2**24)

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel = 3, thresh=(0, 255)):
	    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	    if orient == 'x':
	        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	    if orient == 'y':
	        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
	    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	    binary_output = np.zeros_like(scaled_sobel)
	    binary_output[(scaled_sobel >= thresh[0]) * (scaled_sobel <= thresh[1])]
	    return binary_output

    def mag_thresh(self, image, sobel_kernel=3, mag_thresh=(0, 255)):
	    gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	    gradmag = np.sqrt(sobelx**2 + sobely**2)
	    scale_factor = np.max(gradmag)/255
	    gradmag = (gradmag/scale_factor).astype(np.uint8)
	    binary_output = np.zeros_like(gradmag)
	    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
	    return binary_output

    def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):
	    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	    with np.errstate(divide='ignore', invalid='ignore'):
		    absgraddir = np.absolute(np.arctan(sobely/sobelx))
		    binary_output = np.zeros_like(absgraddir)
		    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
	    return binary_output

    def color_threshold(self, image, sthresh=(0,255), vthresh=(0,255), lthresh=(0,255)):
	    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	    s_channel = hls[:,:,2]
	    s_binary = np.zeros_like(s_channel)
	    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

	    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	    v_channel = hsv[:,:,2]
	    v_binary = np.zeros_like(v_channel)
	    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

	    luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
	    l_channel = luv[:,:,2]
	    l_binary = np.zeros_like(l_channel)
	    l_binary[(l_channel >= lthresh[0]) & (l_channel <= lthresh[1])] = 1

	    output = np.zeros_like(s_channel)
	    output[(s_binary == 1) & (v_binary == 1) & (l_binary == 1)] = 1
	    return output

    def window_mask(self, width, height, img_ref, center, level):
	    output = np.zeros_like(img_ref)
	    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
	    return output

    def camera_callback_compressed(self, data):
        compressed = data.data
        np_arr = np.fromstring(data.data, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        img = image

        preprocessImage = np.zeros_like(img[:,:,0])
        gradx = self.abs_sobel_thresh(img, orient='x', thresh=(12,255))
        grady = self.abs_sobel_thresh(img, orient='y', thresh=(25,255))
        c_binary = self.color_threshold(img, sthresh=(50,255), vthresh=(100,255), lthresh=(50,255))
        m_binary = self.mag_thresh(img, sobel_kernel=3, mag_thresh=(0,25))
        d_binary = self.dir_threshold(img, sobel_kernel=15, thresh=(0.0,1.5))
        preprocessImage[((gradx == 1) & (grady == 1) | (m_binary == 1) & (d_binary == 1) & (c_binary == 1))] = 255

        cv2.imshow('pre', preprocessImage)
        cv2.waitKey(1)

        img_size = (img.shape[1], img.shape[0])
        bot_width = 0.76
        mid_width = 0.08
        height_pct = 0.62
        bottom_trim = 0.935
        src = np.float32([[img.shape[1]*(0.5-mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(0.5+mid_width/2),img.shape[0]*height_pct],
            [img.shape[1]*(0.5+bot_width/2),img.shape[0]*bottom_trim], [img.shape[1]*(0.5-bot_width/2),img.shape[0]*bottom_trim]])
        offset = img_size[0]*0.80
        dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(preprocessImage,M,img_size,flags=cv2.INTER_LINEAR)

        window_width = 25
        window_height = 88

        curve_centers = tracker(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin = 25, My_ym = 10/720, My_xm = 4/384, Mysmooth_factor = 15)

        window_centroids = curve_centers.find_window_centroids(warped)

        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        rightx = []
        leftx = []

        for level in range(0, len(window_centroids)):
            leftx.append(window_centroids[level][0])
            rightx.append(window_centroids[level][1])

            l_mask = self.window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = self.window_mask(window_width,window_height,warped,window_centroids[level][1],level)

            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        template = np.array(r_points+l_points,np.uint8)
        zero_channel = np.zeros_like(template)
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8)
        warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8)
        result = cv2.addWeighted(warpage,1,template,0.5,0.0)

        yvals = np.linspace(0, 719, num=720)
        res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)

        left_fit = np.polyfit(res_yvals, leftx, 2)
        left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
        left_fitx = np.array(left_fitx,np.int32)

        right_fit = np.polyfit(res_yvals, rightx, 3)
        right_fitx = right_fit[0]*yvals*yvals*yvals + right_fit[1]*yvals*yvals + right_fit[2]*yvals + right_fit[3]
        righ_fitx = np.array(right_fitx,np.int32)
    
        left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2),axis=0), np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
        right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0), np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
        middel_marker = np.array(list(zip(np.concatenate((left_fitx+window_width/2,right_fitx[::-1]-window_width/2),axis=0), np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

        road = np.zeros_like(img)
        road_bkg = np.zeros_like(img)

        cv2.fillPoly(road, [left_lane], color=[255,0,0])
        cv2.fillPoly(road, [right_lane], color=[0,0,255])
        cv2.fillPoly(road, [middel_marker], color=[0,255,0])
        cv2.fillPoly(road_bkg, [left_lane], color=[255,255,255])
        cv2.fillPoly(road_bkg, [right_lane], color=[255,255,255])

        road_warped = cv2.warpPerspective(road,Minv,img_size,flags=cv2.INTER_LINEAR)
        road_warped_bkg = cv2.warpPerspective(road_bkg,Minv,img_size,flags=cv2.INTER_LINEAR)

        base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
        result = cv2.addWeighted(base, 1.0, road_warped, 1.0, 0.0)
            
        cv2.imshow('res', result)
        cv2.waitKey(1)
