/****** RANSAC 提取直线 ******/
/****** RANSAC 提取有角度约束的直线 设置坐标轴和坐标轴的夹角 ******/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include "../utils/pcd_viewer.h"

int main(int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input (new pcl::PointCloud<pcl::PointXYZ>);
    uint32_t line_size(0);
    uint32_t remain_size(0);
    pcl::io::loadPCDFile("bed_0003.pcd",*cloud_input);

    /****** 创建拟合模型 ******/
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);  // 拟合点云的索引
    pcl::SACSegmentation<pcl::PointXYZ> seg;   // 分割器

    seg.setOptimizeCoefficients(true);      // 可选择配置，设置模RANSAC型系数需要优化
    seg.setModelType(pcl::SACMODEL_LINE);   // 拟合目标形状 line
    /****** 拟合有角度约束的直线 ******/
    // seg.setModelType(pcl::SACMODEL_PARALLEL_LINE);  // 设置模型类型为：有方向约束的直线拟合
    // const Eigen::Vector3f axis(1, 0, 0);
	// const double eps = 1.2;
    // seg.setAxis(axis);    // 设置坐标轴
	// seg.setEpsAngle(eps); // 设置与坐标轴的夹角
    seg.setMethodType(pcl::SAC_RANSAC);     // 拟合方法：随机采样法
    seg.setDistanceThreshold(0.05);          // 设置误差容忍范围，也就是阈值，直线模型的 宽度
    seg.setMaxIterations(500);              // 最大迭代次数，默认迭代50次
    seg.setInputCloud(cloud_input);         // 输入点云
    seg.segment(*inliers, *coefficients);   // 拟合点云

    /****** 拟合的模型系数 ******/
    // 三维直线，6 个系数，分别是 x0 y0 z0 ，m n p
    // (x-x0)/m = (y-y0)/n = (z-z0)/p
    std::cout << "a: " << coefficients->values[0] << endl;
    std::cout << "b: " << coefficients->values[1] << endl;
    std::cout << "c: " << coefficients->values[2] << endl;
    std::cout << "d: " << coefficients->values[3] << endl;
    std::cout << "e: " << coefficients->values[4] << endl;
    std::cout << "f: " << coefficients->values[5] << endl;


    /****** 索引拟合模型 ******/
    pcl::PointCloud<pcl::PointXYZ>::Ptr line(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr remain(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_input); // 设置输入点云
    extract.setIndices(inliers);        // 设置索引
    extract.setNegative(false);         // false提取索引内点, true提取外点
    extract.filter(*line);
    extract.setNegative(true);
    extract.filter(*remain);
    
    /****** view ******/
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_all (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointXYZRGB point_rgb;
    line_size = line -> size();
    remain_size = remain -> size();
    rgb_all -> width = (line_size + remain_size);
    rgb_all -> height = 1;
    rgb_all -> resize(line_size + remain_size);
    for(uint32_t i = 0; i<line_size; i++)
    {
        point_rgb.x = line -> points[i].x;
        point_rgb.y = line -> points[i].y;
        point_rgb.z = line -> points[i].z;
        point_rgb.r = 255;
        point_rgb.g = 0;
        point_rgb.b = 0;
        rgb_all -> push_back(point_rgb);
    }
    for(uint32_t k(0); k<remain_size; k++)
    {
        point_rgb.x = remain -> points[k].x;
        point_rgb.y = remain -> points[k].y;
        point_rgb.z = remain -> points[k].z;
        point_rgb.r = 0;
        point_rgb.g = 0;
        point_rgb.b = 255;
        rgb_all -> push_back(point_rgb);
    }



    cloud_viewer(rgb_all, 0);
    return 0;
}


