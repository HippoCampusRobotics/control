cmake_minimum_required(VERSION 3.0.2)
project(control)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  hippocampus_common
  hippocampus_msgs
  path_planning
  dynamic_reconfigure
  mavros_msgs
)

catkin_python_setup()

## Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
  config/PidControl.cfg
  config/CarrotControl.cfg
)

catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  nodes/attitude_mixer_node
  nodes/depth_control_node
  nodes/path_follower_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
