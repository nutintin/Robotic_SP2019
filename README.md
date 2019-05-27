# Hand-Based Robot-Human Interaction
This project is the final project of our Robotics course 2019 in Università della Svizzera italiana. We are implementing hand-based Robot-Human interaction. The general idea of this project is to make Robot ( Mighty Thymio ) can interact with environment and humans. Moreover, in this project, the interaction that we present is interaction of human hand which is detected by Mighty Thymio. While hand is detected Mighty Thymio can be able to follow the movement of it. The movement are rotating to the right or left and moving front or back while hand is detected in the middle.


### Prerequisites

What things you need to install the software and how to install them

First we need to install ROS, you can follow this [link](http://wiki.ros.org/Installation/Ubuntu).
And also installation

Then we need to change default tensorflow first to avoid error tensorflow
```
python2.7 -m pip install https://storage.^Cogleapis.com/tensorflow/linux/cpu/tensorflow-1.9.0-cp27-none-linux_x86_64.whl
```

and install requirements.txt

```
pip install -r requirements.txt
```

## Deployment

go to src folder in your project folder
```
[your_directory_project_folder]/src
example : '~/catkin_ws/src/hand_following/src/'
```
run the main.py file
```
python2 main.py
```

## Authors

* **Amirehsan Davoodi** - [AmirDavoodi](https://github.com/AmirDavoodi)
* **Weimeng Pu** - [weimengpu](https://github.com/weimengpu)
* **Reinard Lazuardi Kuwandy** - [nutintin](https://github.com/nutintin)

## Acknowledgments

* https://github.com/victordibia/handtracking - handtracking
* https://github.com/jeguzzi/mighty-thymio - mighty thymio
* https://github.com/FrancescoSaverioZuppichini/Robotics-Final-Project - object detection with mighty thymio
* https://github.com/Mirko-Nava/Learning-Long-range-Perception - Learning Long-range Perception using Self-Supervision from Short-Range Sensors and Odometry


