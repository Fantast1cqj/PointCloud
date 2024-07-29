# Point Cloud
- [Point Cloud](#point-cloud)
  - [Lidar 运动补偿](#lidar-运动补偿)
    - [livox CustomMsg 格式点云预处理](#livox-custommsg-格式点云预处理)
  - [点云配准](#点云配准)
  - [3D 语义分割](#3d-语义分割)

##  Lidar 运动补偿
https://blog.csdn.net/brightming/article/details/118250783

https://blog.csdn.net/qq_30460905/article/details/124919036

代码：~/PointCloud/PointCloud_ws/src/motion_compensation

### livox CustomMsg 格式点云预处理
**加载 livox 头文件问题**
    
    cpp：
    #include <livox_ros_driver/CustomMsg.h>

    CMakeLists.txt:
    find_package(catkin REQUIRED COMPONENTS
    livox_ros_driver
    )

    package.xml:
    <build_depend>livox_ros_driver</build_depend>
    <exec_depend>livox_ros_driver</exec_depend>







##  点云配准

##  3D 语义分割
![alt text](image.png)
![alt text](image-1.png)

分类 目标检测 语义分割 区别
语义分割给每个像素一个 label
