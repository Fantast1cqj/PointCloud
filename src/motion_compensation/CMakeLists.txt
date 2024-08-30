cmake_minimum_required(VERSION 2.8.3)
project(motion_compensation_node)

set(CMAKE_BUILD_TYPE "Release")
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")


find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  nav_msgs
  livox_ros_driver
)

message(STATUS "livox_ros_driver_DIR = ${livox_ros_driver_DIR}")
message(STATUS "livox_ros_driver_INCLUDE_DIRS = ${livox_ros_driver_INCLUDE_DIRS}")
message(STATUS "livox_ros_driver_LIBS = ${livox_ros_driver_LIBS}")


# set(livox_ros_driver_DIR "/home/taotao/code/vslam/livox_driver_ws/devel/share/livox_ros_driver/cmake/")

catkin_package()
# catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ego_planner
#  CATKIN_DEPENDS plan_env path_searching bspline_opt traj_utils 
# #  DEPENDS system_lib
# )


include_directories(
  include
  SYSTEM
  ${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)


add_executable(motion_comp src/motion_comp_node.cpp src/preprocess.cpp)




target_link_libraries(motion_comp
  ${catkin_LIBRARIES}
)
