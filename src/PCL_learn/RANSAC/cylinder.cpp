#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include "../utils/pcd_viewer.h"

using namespace std;

int
main(int argc, char** argv)
{
	//------------------------------读取点云数据---------------------------------
	pcl::PointCloud<pcl::PointXYZ >::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ >);
	pcl::PCDReader reader;
	reader.read("airplane_0002.pcd", *cloud_in);
	cout << "PointCloud has: " << cloud_in->points.size() << " data points." << endl;

	
	//--------------------------------计算法线-----------------------------------
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ >::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ >);
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	n.setSearchMethod(tree);
	n.setInputCloud(cloud_in);
	n.setKSearch(50);
	n.compute(*normals);

	//------------------------------创建分割对象---------------------------------
	pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
	pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients), coefficients_cylinder(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices), inliers_cylinder(new pcl::PointIndices);

	// ------------------------点云分割，提取平面上的点--------------------------
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
	seg.setNormalDistanceWeight(0.1);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(500);
	seg.setDistanceThreshold(0.05);
	seg.setInputCloud(cloud_in);
	seg.setInputNormals(normals);
	seg.segment(*inliers_plane, *coefficients_plane);//获取平面模型系数和平面上的点
	cout << "Plane coefficients: " << *coefficients_plane << endl;
	//----------------------------------提取平面---------------------------------
	pcl::ExtractIndices<pcl::PointXYZ > extract;
	extract.setInputCloud(cloud_in);
	extract.setIndices(inliers_plane);
	extract.setNegative(false);
	pcl::PointCloud<pcl::PointXYZ >::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ >());
	extract.filter(*cloud_plane);
	cout << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points." << endl;

	//-------------------------------提取圆柱体模型------------------------------
	pcl::ExtractIndices<pcl::Normal> extract_normals;
	pcl::PointCloud<pcl::PointXYZ >::Ptr cloud_filtered2(new pcl::PointCloud<pcl::PointXYZ >);
	pcl::PointCloud<pcl::Normal>::Ptr normals2(new pcl::PointCloud<pcl::Normal>);
	//获取平面以外的点和点的法线
	extract.setNegative(true);
	extract.filter(*cloud_filtered2);
	extract_normals.setNegative(true);
	extract_normals.setInputCloud(normals);
	extract_normals.setIndices(inliers_plane);
	extract_normals.filter(*normals2);
	//为圆柱体分割创建分割对象，并设置参数
	seg.setOptimizeCoefficients(true);        //设置对估计的模型系数需要进行优化
	seg.setModelType(pcl::SACMODEL_CYLINDER); //设置分割模型为圆柱型
	seg.setMethodType(pcl::SAC_RANSAC);       //设置采用RANSAC作为算法的参数估计方法
	seg.setNormalDistanceWeight(0.1);         //设置表面法线权重系数
	seg.setMaxIterations(500);               //设置迭代的最大次数
	seg.setDistanceThreshold(0.02);           //设置内点到模型的距离允许最大值 
	seg.setRadiusLimits(0, 0.1);              //设置估计出圆柱模型的半径范围
	seg.setInputCloud(cloud_filtered2);
	seg.setInputNormals(normals2);
	//获取圆柱模型系数和圆柱上的点
	seg.segment(*inliers_cylinder, *coefficients_cylinder);
	cout << "Cylinder coefficients: " << *coefficients_cylinder << endl;


    // 提取圆柱点
    pcl::ExtractIndices<pcl::PointXYZ > extract_cylinder;
    extract.setInputCloud(cloud_filtered2);
	extract.setIndices(inliers_cylinder);
	extract.setNegative(false);
	pcl::PointCloud<pcl::PointXYZ >::Ptr cloud_cylinder(new pcl::PointCloud<pcl::PointXYZ >());
	extract.filter(*cloud_cylinder);
	cout << "PointCloud representing the cylinder component: " << cloud_cylinder->points.size() << " data points." << endl;

    cloud_viewer(cloud_cylinder, 0);

	return 0;
}

