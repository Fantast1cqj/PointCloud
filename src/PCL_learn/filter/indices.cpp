/****** 点云索引提取器 ******/
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <boost/thread/thread.hpp>
#include "../utils/pcd_viewer.h"


int main(int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_output(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile("airplane_0002.pcd",*cloud_input);

    /****** 添加索引 ******/
    pcl::PointIndicesPtr inds(new pcl::PointIndices);  // 也可以用智能指针
    // pcl::PointIndices indices;   // 配合 29 行使用
    uint16_t i = 0;
    for(i = 0; i < 50; i++)
    {
        inds -> indices.push_back(i);
    }
    

    pcl::ExtractIndices<pcl::PointXYZ> extr; // 索引提取器
    extr.setInputCloud(cloud_input);           // 设置输入点云
    extr.setIndices(inds);
    // extr.setIndices(boost::make_shared<const pcl::PointIndices>(indices)); // 当 indices 为一个变量不是指针时使用 设置索引 创建一个共享智能指针
    extr.filter(*cloud_output);     // 提取出 indices 中的点云

    cloud_viewer(cloud_output, 1);

    return 0;
}