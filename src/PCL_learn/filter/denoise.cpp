/****** 点云去噪 ******/
// 点云半径滤波：设置半径，设置点数 n，点为中心，球中点数少于 n 则去除该点

// 统计滤波器：遍历所有点，取某个点周围 k 个点，算 k 个距离，并计算距离的均值和方差，保留 (μ - std * σ, μ + std * σ) 距离内的点

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl/filters/radius_outlier_removal.h>  //半径滤波器
#include <pcl/filters/statistical_outlier_removal.h> // 统计滤波
#include "../utils/pcd_viewer.h"

int main(int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr radius_output (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filter1_output (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile("airplane_0002.pcd",*cloud_input);

    /****** 点云半径滤波 ******/
    pcl::StopWatch time;
    time.reset();
    pcl::RadiusOutlierRemoval<pcl::PointXYZ>::Ptr filter (new pcl::RadiusOutlierRemoval<pcl::PointXYZ>);
    filter -> setInputCloud(cloud_input);
    filter -> setRadiusSearch(0.02);  // 设置半径
    filter -> setMinNeighborsInRadius(5);    // 设置半径内最少点数
    filter -> filter(*radius_output);
    std::cout << "input: " << cloud_input ->size() << std::endl;
    std::cout << "output: " << radius_output ->size() << std::endl;
    std::cout << "time: " << time.getTime() << std::endl;
    // cloud_viewer(radius_output, 1);





    /****** 统计滤波器 ******/
    time.reset();
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> filter_1;
    filter_1.setInputCloud(cloud_input);  // 设置输入点云
    filter_1.setMeanK(50);   // 设置 k 值
    filter_1.setStddevMulThresh(2); // 设置 std 值
    filter_1.filter(*filter1_output);
    std::cout << "input: " << cloud_input ->size() << std::endl;
    std::cout << "output: " << filter1_output ->size() << std::endl;
    std::cout << "time: " << time.getTime() << std::endl;
    cloud_viewer(filter1_output, 1);



    return 0;
}



