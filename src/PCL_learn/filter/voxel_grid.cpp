// 创建 voxel gird 进行下采样，用体素 重心 近似体素内的其他点，比体素中心更慢，但是表示曲面更准确
// Approximate Voxel Grid （中心)
// 改进 Voxel Grid，使用原始点云距离重心最近的点作为下采样的点
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h> // 重心
#include <pcl/filters/approximate_voxel_grid.h> // 中心
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <boost/thread/thread.hpp>





#include <vector>
#include "../utils/pcd_viewer.h"
int main(int argc, char** argv)
{
    /* load data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_output(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_output2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile("airplane_0001.pcd",*cloud_input);






    /****** Voxel Grid （重心) ******/
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud_input);
    vg.setLeafSize(0.05f, 0.05f, 0.05f);
    vg.filter(*cloud_output);

    std::cout << "input point num: " << cloud_input -> size() << std::endl;
    std::cout << "output point num: " << cloud_output -> size() << std::endl;
    // cloud_viewer(cloud_output, 1);






    /****** Approximate Voxel Grid （中心) ******/
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> avf;
    avf.setInputCloud(cloud_input);
    avf.setLeafSize(0.01f, 0.01f, 0.01f);// 最小体素的边长
    avf.filter(*cloud_output2);      // 进行滤波

    std::cout << "input point num: " << cloud_input -> size() << std::endl;
    std::cout << "output point num: " << cloud_output2 -> size() << std::endl;
    // cloud_viewer(cloud_output2, 1);






    /****** 改进 Voxel Grid，使用原始点云距离重心最近的点作为下采样的点 ******/
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_input);

    // pcl::PointIndicesPtr inds = boost::make_shared<pcl::PointIndices>();
    pcl::PointIndicesPtr inds(new pcl::PointIndices);  // 存储点云中点的索引
    // pcl::PointIndices inds;
    
    for(int16_t i = 0; i < cloud_output->size(); i++)
    {
        pcl::PointXYZ searchPoint;
        searchPoint.x = cloud_output->points[i].x;
		searchPoint.y = cloud_output->points[i].y;
		searchPoint.z = cloud_output->points[i].z;

        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);

        if (kdtree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) 
        {
			inds->indices.push_back(pointIdxNKNSearch[0]);
		}
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr final_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud_input, inds->indices, *final_filtered);

    std::cout << "input point num: " << cloud_input -> size() << std::endl;
    std::cout << "output point num: " << final_filtered -> size() << std::endl;




    return 0;
}

