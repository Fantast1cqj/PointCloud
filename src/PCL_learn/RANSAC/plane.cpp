/****** RANSAC 提取平面 ******/
/****** 提取有角度限制的平面 ******/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include "../utils/pcd_viewer.h"
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/common/common_headers.h>

/****** 提取有角度限制的平面 ******/
void get_plane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, float angle, Eigen::Vector3f Axis, pcl::ModelCoefficients::Ptr coefficients, pcl::PointIndices::Ptr inliers)
{
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    
    /****** 分割其配置 ******/
    seg.setOptimizeCoefficients(true);  // 可选择配置，设置模型系数需要优化
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);//设置分割的模型类型
    seg.setMethodType(pcl::SAC_RANSAC);    // 设置所用随机参数估计方法
    seg.setDistanceThreshold(0.05);        // 距离阈值，单位m.
                  
    float EpsAngle= pcl::deg2rad(angle);   // 角度转弧度
    // Eigen::Vector3f Axis(0.0, 0.0, 1.0);
    seg.setAxis(Axis);                     // 指定的轴
    seg.setEpsAngle(EpsAngle);             // 夹角阈值(弧度制)

    seg.setInputCloud(cloud_in);            // 输入点云
    seg.segment(*inliers, *coefficients);  // 存储结果到点集合inliers及存储平面模型系数coefficients
    // if (inliers->indices.size() == 0)
    // {
    //     PCL_ERROR("Could not estimate a planar model for the given dataset.");
    // }


}



int main(int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_output (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile("bed_0003.pcd",*cloud_input);

    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_plane(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_input));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_plane);
    ransac.setDistanceThreshold(0.01); // 设定距离阈值
    ransac.setMaxIterations(500);      // 设置最大迭代次数
    ransac.setProbability(0.99);       // 设置从离群值中选择至少一个样本的期望概率
    ransac.computeModel();             // 拟合平面
    std::vector<int> inliers;
    ransac.getInliers(inliers);          // 获取内点索引
    Eigen::VectorXf coeff;               // 拟合的平面参数
    ransac.getModelCoefficients(coeff);  // 获取拟合平面参数，coeff分别按顺序保存a,b,c,d
    std::cout << "plane: " << coeff[0] << coeff[1] << coeff[2] << coeff[3] << std::endl;


    /****** 获取平面点云 ******/
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud<pcl::PointXYZ>(*cloud_input, inliers, *plane);   // 根据索引提取平面点云，这里的索引是 std::vector<int> 形式
    uint32_t plane_size(0);
    plane_size = plane -> size();
    std::cout << "plane size" << plane_size << std::endl;
    plane_rgb -> width  = plane_size;
    plane_rgb -> height = 1;
    plane_rgb -> resize(plane_size);
    for(int k = 0; k < plane_size; k++)
    {
        plane_rgb -> points[k].x = plane -> points[k].x;
        plane_rgb -> points[k].y = plane -> points[k].y;
        plane_rgb -> points[k].z = plane -> points[k].z;
        plane_rgb -> points[k].r = 0;
        plane_rgb -> points[k].g = 255;
        plane_rgb -> points[k].b = 0;

    }


    /****** 获取平面以外的点云 ******/
    pcl::PointCloud<pcl::PointXYZ>::Ptr remain (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr remain_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);   // 智能指针在 = 赋值的时候需要 resize，pushback()不需要   变量赋值不需要 resize
    pcl::PointIndicesPtr inds (new pcl::PointIndices);
    uint32_t model_pt_num = inliers.size();
    uint32_t k(0);
    for(k = 0; k < model_pt_num; k++)
    {
        inds->indices.push_back(inliers[k]);
    }
    pcl::ExtractIndices<pcl::PointXYZ> extr; // 索引提取器
    extr.setInputCloud(cloud_input);         // 设置输入点云
    extr.setIndices(inds);                   // 设置索引，这里的索引是 pcl::PointIndicesPtr 形式
    extr.setNegative(true);                  // 提取索引的外点
    extr.filter(*remain);     // 提取出 indices 中的点云
    uint32_t remain_size(0);
    remain_size = remain -> size();
    std::cout << "remain size" << remain_size << std::endl;
    remain_rgb -> width  = remain_size;
    remain_rgb -> height = 1;
    remain_rgb -> resize(remain_size);
    for(int k = 0; k < plane_size; k++)
    {
        remain_rgb -> points[k].x = remain -> points[k].x;
        remain_rgb -> points[k].y = remain -> points[k].y;
        remain_rgb -> points[k].z = remain -> points[k].z;
        remain_rgb -> points[k].r = 255;
        remain_rgb -> points[k].g = 255;
        remain_rgb -> points[k].b = 255;

    }

    *cloud_output = *plane_rgb + *remain_rgb;
    // cloud_viewer(cloud_output, 0);





    /****** 提取有角度限制的平面 ******/
    Eigen::Vector3f Axis(0.0, 0.0, 1.0);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers2(new pcl::PointIndices);

    get_plane(cloud_input, 20, Axis, coefficients, inliers2);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_input);
    extract.setIndices(inliers2);
    extract.filter(*cloud_filtered);

    cloud_viewer(cloud_filtered, 0);


    return 0;
}





