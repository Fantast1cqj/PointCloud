/****** PCL 直通滤波器 pcl::PassThrough<pcl::PointXYZ> filter ******/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include "../utils/pcd_viewer.h"

int main(int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_output (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile("bed_0003.pcd",*cloud_input);

    /****** 初始化滤波器 ******/
    pcl::PassThrough<pcl::PointXYZ> filter;
    filter.setInputCloud(cloud_input);
    filter.setFilterFieldName("z");    // 设置滤波字段
    filter.setFilterLimits(0.0, 1.5);  // 设置过滤范围
    // filter.setKeepOrganized(true);     // 保持有序点云结构 在有序点云上应用
    filter.setNegative(false);         // 保留还是过滤范围内的点 false保留 true删除范围内的点
    filter.filter(*cloud_output);

    // cloud_viewer(cloud_output, 1);




    /****** 过滤得到点的 index ******/
	pcl::PassThrough<pcl::PointXYZ> pass(true); // 用true初始化将允许我们提取被删除的索引
	pass.setInputCloud(cloud_input);
	pass.setFilterFieldName("x");
	pass.setFilterLimits(0.0, 0.5);
	pass.setNegative(false);       // 设置 false 则保留范围内的点
	std::vector<int> indices_x;
	pass.filter(indices_x);        // indices_x 保存的是在(0，0.5)范围点的索引

    std::cout << indices_x[1] << std::endl;

    pcl::IndicesConstPtr indices_rem;
	indices_rem = pass.getRemovedIndices(); // 获取被剔除点的索引

    std::cout <<  indices_x.size() << std::endl;    // 
    std::cout <<  indices_rem->size() << std::endl;

	// pcl::IndicesPtr  index_ptr_x = std::make_shared<std::vector<int>>(indices_x);

	// pass.setIndices(index_ptr_x);
	// pass.setFilterFieldName("z");
	// pass.setFilterLimits(0.0, 7.0);
	// pass.setNegative(true);

	// std::vector<int> indices_xz;
	// pass.filter(indices_xz);


}



