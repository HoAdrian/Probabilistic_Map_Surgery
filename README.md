# Installation and Documentation
## Conda environments ready to use:
* rlgpu.yml: for collecting data in Isaac Gym Simulation and training models
* py39: miscellaneous procedures such as creating meshes and urdf

## Isaac Gym and ROS
* Install Isaac Gym: Carefully follow the official installation guide and documentation from Isaac Gym. You should be able to find the documentation on isaacgym/docs/index.html. Make sure you select NVIDIA GPU before running the examples.
* Install ROS Melodic (system-wide installation)

## Other dependencies
* Set up:
```sh
# Step 1: Create a catkin workspace called catkin_ws: http://wiki.ros.org/catkin/Tutorials/create_a_workspace

# Step 2: Clone this repo, put the folders in this repo into the src folder
cd src
git clone <github link of this repo>

# Step 3: Clone other repos into the src folder:
git clone https://github.com/eric-wieser/ros_numpy.git
git clone https://github.com/gt-ros-pkg/hrl-kdl.git
git clone https://baotruyenthach@bitbucket.org/robot-learning/point_cloud_segmentation.git
```
## to run the motion planner for moving the robot in simulation.
roslaunch shape_servo_control dvrk_isaac.launch. Remember to source the workspace (source ~/catkin_ws/devel/setup.bash) before running any ros code. 

## code for the current project
* the code in the active_perception folder is for the current project. run.sh contains example commands to run. 


