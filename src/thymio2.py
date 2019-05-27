import numpy as np
import cv2
import sys, datetime
import rospy, rosnode, rostopic, rosmsg
import std_msgs.msg, sensor_msgs.msg, nav_msgs.msg
from sensor_msgs.msg import Range, Image, CompressedImage
from threading import Thread
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from PID import PID
from utils import detector_utils as detector_utils

Params = detector_utils.Params

detection_graph, sess = detector_utils.load_inference_graph()

# callback sensors
def callback(fun,*args, **kwargs):
    def delay(x):
        fun(x, *args, **kwargs)
    return delay

class Thymio:
    
    def __init__(self, thymio_name, hooks=[]):
        """init"""
        self.thymio_name = thymio_name

        self.frame = []
        self.hooks = hooks
        self.sensors_cache_values = np.zeros(7)
        self.sensors_cache = {}
        self.time_elapsed = 0
        self.start = 0

        rospy.init_node('hand_following_thymio_controller', anonymous=True)

        self.velocity_publisher = rospy.Publisher(self.thymio_name + '/cmd_vel',
                                                  Twist, queue_size=10)

        self.pose_subscriber = rospy.Subscriber(self.thymio_name + '/odom',
                                                Odometry, self.update_state)

        # angular pid
        self.angular_pid = PID(Kd=5, Ki=0, Kp=0.1)
        self.linear_pid = PID(Kd=5, Ki=0, Kp=0.1)
        self.object_pid = PID(Kd=3, Ki=0, Kp=0.1)

        # sensors subscriber
        self.sensors_names = ['right', 'center_right', 'center', 'center_left', 'left', 'rear_right', 'rear_left']

        self.sensors_subscribers = [rospy.Subscriber(self.thymio_name + '/proximity/' + sensor_name,Range,
                                                     callback(self.sensors_callback,i,sensor_name)) 
                                                     for i,sensor_name in enumerate(self.sensors_names)]

        self.stopped = False
        self.counter = 0

        # create camera subscriber
        self.camera_subscriber = rospy.Subscriber(self.thymio_name + '/camera/image_raw/compressed', CompressedImage, self.camera_callback_compressed, queue_size=1, buff_size=2**24)
        
        self.rate = rospy.Rate(10)
        self.hand_buffer = [None,None,None]

        self.obstacle = None
    
    def update_vel(self, linear, angular):
        """ Update velocity of mighty thymio"""
        vel_msg = Twist()

        vel_msg.linear.x = linear.x
        vel_msg.linear.y = linear.y
        vel_msg.linear.z = linear.z

        vel_msg.angular.x = angular.x
        vel_msg.angular.y = angular.y
        vel_msg.angular.z = angular.z

        self.vel_msg = vel_msg
    
    def move(self, linear, angular):
        """Moves the migthy thymio"""
        self.update_vel(linear, angular)

        while not rospy.is_shutdown():
            self.velocity_publisher.publish(self.vel_msg)
            self.rate.sleep()

        rospy.spin()

    def stop(self):
        """ Stop mighty thymio """
        self.update_vel(Params(), Params())

    def sensors_callback(self, data, sensor_id, name):
        """ sensors callback """
        self.sensors_cache[name] = data

        try:
            self.on_receive_sensor_data(data, sensor_id, name)
            for hook in self.hooks:
                hook.on_receive_sensor_data(self, data, sensor_id, name)
        except KeyError:
            pass

    def on_receive_sensor_data(self, data, sensor_id, name):
        """ sensors receive data """
        val = data.range
        max = data.max_range

        if(val == np.inf): val = 0

        else:
            if(val < 0): val = data.min_range
            val = max - val
            val = val / max

        if sensor_id >= 5: val *= -1

        self.sensors_cache_values[sensor_id] = val

        self.obstacle = np.sum(self.sensors_cache_values) != 0

        if self.obstacle:
            self.hand_buffer = [None, None, None]
            lin_err = np.sum(self.sensors_cache_values) / self.sensors_cache_values.shape[0]
            ang_err = np.sum(self.sensors_cache_values[:2] - self.sensors_cache_values[3:5]) +  (self.sensors_cache_values[5] - self.sensors_cache_values[6])

            ang_vel = self.angular_pid.step(ang_err, 0.1)
            vel = self.linear_pid.step(lin_err, 0.1)

            self.last_elapsed_sensors = self.time_elapsed

            self.update_vel(Params(x=-vel), Params(z=-ang_vel))
    
    def update_state(self, data):
        """A new Odometry message has arrived. See Odometry msg definition."""
        # Note: Odmetry message also provides covariance
        self.current_pose = data.pose.pose
        self.current_twist = data.twist.twist
        quat = (
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w)
        (roll, pitch, yaw) = euler_from_quaternion (quat)

        # rospy.loginfo("State from Odom: (%.5f, %.5f, %.5f) " % (self.current_pose.position.x, self.current_pose.position.y, yaw))
        self.time_elapsed += 10

        for hook in self.hooks:
            hook.on_update_pose(self)
    
    def camera_callback_compressed(self, data):
        """ camera callback """ 
        start_time = datetime.datetime.now()
        
        # get the camera info height and width
        format_msg = rospy.wait_for_message('/thymio10/camera/camera_info', sensor_msgs.msg.CameraInfo)

        # create a window
        cv2.namedWindow('Hand Detection')#, cv2.WINDOW_NORMAL)

        # extract data from mighty thymio camera
        compressed = data.data
        np_arr = np.fromstring(data.data, dtype=np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        self.draw_image(image_np, image_np.shape[1], image_np.shape[0], start_time)

    def draw_image(self, image_np, im_width, im_height, start_time):
        """ draw image """

        # init
        num_frames = 0
        num_hands_detect = 1
        scores_thresh = 0.5
        
        
        ## (haven't tried) purpose to avoid go straight or move keep go
        vel = 0.0
        ang_vel = 0.0
        
        ## calculate the mid point
        mid_point = im_width // 2

        ## create boxes and detect hand
        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

        ## draw the box of the hand if it was detected
        hand_buffer, width_cur = detector_utils.draw_box_on_image(num_hands_detect, 0.15,
                                         scores, boxes, im_width, im_height,
                                         image_np, mid_point, self.hand_buffer)
        
        ## update the hand buffer value and ang_err value
        self.hand_buffer = hand_buffer
        ang_err = self.hand_buffer[2]

        print("width_cur = " + str(width_cur))
        
        ## if hand detected
        if ang_err is not None:
            if ang_err < -70  or  ang_err > 70:
                ang_vel = ang_err / im_width
                vel = 0
            ## angular error /position of the and is in the middle -> go straight
            elif ang_err > -70 or ang_err < 70:
                ang_vel = 0
                ## size box / position of hand is to close with camera -> move back
                if width_cur >= 170:
                    vel = -0.07
                ## size box / position of hand is in mid range -> stop
                elif width_cur > 150 and width_cur < 170:
                    vel = 0.00
                ##(haven't tried) size box / position of hand is far from the camera go front
                elif width_cur <= 150:
                    vel = 0.07
                ## other case stay
                # else:
                #     vel = 0.00
            else:
                vel = 0
                ang_vel = 0

        ## otherwise
        else:
            ang_vel = 0
            vel = 0
        
        ## update velocity linear and angular 
        self.update_vel(Params(x=vel), Params(z=-ang_vel))

        ## calculate fps
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time
        
        ## draw fps in window
        detector_utils.draw_fps_on_image("fps : " + str(int(fps)),
                                                 image_np)
        ## show stream
        cv2.imshow('Hand Detection',
                    cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
