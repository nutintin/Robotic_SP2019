#! /usr/bin/env python3

from thymio2 import Thymio
from utils import detector_utils
Params = detector_utils.Params

import rospy
import cv2

thymio = Thymio('thymio10')

thymio.move(Params(0),Params())

# while not rospy.is_shutdown():
#    rospy.sleep(1.)