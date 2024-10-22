#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
// #include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include "../utils/pcd_viewer.h"

int main()
{
	// ------------------------------加载点云数据------------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("airplane_0002.pcd", *cloud_input) < 0)
	{
		PCL_ERROR("Couldn't read file \n");
		return -1;
	}

	// -------------------------------拟合球体--------------------------------
	pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr model(new pcl::SampleConsensusModelSphere<pcl::PointXYZ>(cloud_input));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_output(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model);	// 定义RANSAC算法对象
	ransac.setDistanceThreshold(0.01);		   // 设置距离阈值
	ransac.setMaxIterations(500);			   // 设置最大迭代次数
	ransac.computeModel();
	Eigen::VectorXf coeff;
	ransac.getModelCoefficients(coeff);		   // 球参数
	std::vector<int> ranSacInliers;            // 获取属于拟合出的内点
	ransac.getInliers(ranSacInliers);
	
	pcl::copyPointCloud(*cloud_input, ranSacInliers, *cloud_output);

	std::cout << "x: " << coeff[0] << "\n y: " << coeff[1] << "\n z: " << coeff[2]
		<< "\n r: " << coeff[3]
		<< std::endl;



    cloud_viewer(cloud_output, 0);

	return (0);
}
