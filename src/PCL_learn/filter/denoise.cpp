/****** 点云去噪 ******/
// 点云半径滤波：设置半径，设置点数 n，点为中心，球中点数少于 n 则去除该点

// 统计滤波器：遍历所有点，取某个点周围 k 个点，算 k 个距离，并计算距离的均值和方差，保留 (μ - std * σ, μ + std * σ) 距离内的点

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl/filters/radius_outlier_removal.h>  //半径滤波器
#include <pcl/filters/statistical_outlier_removal.h> // 统计滤波
#include <pcl/filters/convolution_3d.h>  // 高斯滤波
#include "../utils/pcd_viewer.h"

int main(int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input_2 (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointCloud<pcl::PointXYZ>::Ptr radius_output (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filter1_output (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr gassFilter_output(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile("airplane_0002.pcd",*cloud_input);
    pcl::io::loadPCDFile("rabbit_GS.pcd",*cloud_input_2);

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
    // cloud_viewer(filter1_output, 1);






    /****** Gaussian 滤波器 ******/
    // 对每个点周围的点乘一个 Gaussian 权重
    // https://blog.csdn.net/qq_36686437/article/details/114160482
    // https://blog.csdn.net/Man_1man/article/details/130167933

    // 设置 Gaussian Kernel
    pcl::filters::GaussianKernel<pcl::PointXYZ, pcl::PointXYZ> kernel;
    kernel.setSigma(4);       // Gaussian 标准差，决定函数的宽度
    kernel.setThresholdRelativeToSigma(4); // 考虑 4*Sigma 以内的点
    kernel.setThreshold(0.5);    // 平cloud_input滑过程中考虑的点的最大距离（欧式距离） 超过 0.05 不在考虑范围 

    // 设置 kdtree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree (new pcl::search::KdTree<pcl::PointXYZ>);
    kdtree -> setInputCloud(cloud_input);

    // 设置 convolution
    pcl::filters::Convolution3D<pcl::PointXYZ, pcl::PointXYZ, pcl::filters::GaussianKernel<pcl::PointXYZ, pcl::PointXYZ>> convolution;
    convolution.setKernel(kernel);
    convolution.setInputCloud(cloud_input);
    convolution.setNumberOfThreads(8);   // 8 线程进行卷积
    convolution.setSearchMethod(kdtree);
    convolution.setRadiusSearch(0.5);
    convolution.convolve(*gassFilter_output);
    cloud_viewer(cloud_input, 1);


    return 0;
}