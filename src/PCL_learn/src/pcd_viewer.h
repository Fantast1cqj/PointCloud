#ifndef MY_HEADER_H
#define MY_HEADER_H

# include <iostream>
# include <pcl/point_cloud.h>
# include <pcl/io/pcd_io.h>
# include <pcl/point_types.h>

#include <thread>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_representation.h>
#include <pcl/common/transforms.h>

void cloud_viewer_simple (pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
void cloud_viewer(pcl::PointCloud<pcl::PointNormal>::ConstPtr cloud, u_int8_t mod);
void cloud_viewer(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, u_int8_t mod);
void cloud_viewer (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals,u_int8_t mod);





#endif