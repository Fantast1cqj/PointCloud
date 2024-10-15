#include <iostream>
#include <pcl/io/pcd_io.h>                    
#include <pcl/point_types.h>                         
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <boost/thread/thread.hpp>
#include "../utils/pcd_viewer.h"



void fitMultiLines(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::vector<pcl::PointCloud<pcl::PointXYZ>>& models_points, std::vector<pcl::ModelCoefficients>& models_arg, uint16_t models_num)
{
	
	
    pcl::SACSegmentation<pcl::PointXYZ> seg; // 分割器
	seg.setOptimizeCoefficients(true);       // 可选择配置，设置模RANSAC型系数需要优化
    seg.setModelType(pcl::SACMODEL_LINE);    // 拟合目标形状 line 
    seg.setMethodType(pcl::SAC_RANSAC);      // 拟合方法：随机采样法
    seg.setDistanceThreshold(0.03);          // 设置误差容忍范围，也就是阈值，直线模型的 宽度
    seg.setMaxIterations(500);               // 最大迭代次数，默认迭代50次
	// pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_copy(cloud_in);

	
	for(uint16_t i(0); i < models_num; i++)
	{	
		pcl::PointIndices::Ptr Indexes(new pcl::PointIndices);  // 拟合点云的索引
		pcl::PointCloud<pcl::PointXYZ>::Ptr remain (new pcl::PointCloud<pcl::PointXYZ>);
		pcl::ModelCoefficients::Ptr arg(new pcl::ModelCoefficients);   // 拟合模型的参数
		std::cout << "remain points" << cloud_in->size() << std::endl;
		seg.setInputCloud(cloud_in);  	// input

		seg.segment(*Indexes, *arg);  	// 拟合点云

		std::cout << "size of Indexes" << Indexes->indices.size() << std::endl;

		
		if(Indexes -> indices.size() > 20)
		{
		
			models_arg.push_back(*arg);


			// 不提取索引，直接将 cloud_in 中 Indexes 对应的点设置为NaN

			pcl::ExtractIndices<pcl::PointXYZ> extract; // 索引提取器
			extract.setInputCloud(cloud_in);    // 设置输入点云
			extract.setIndices(Indexes);        // 设置索引
			extract.setNegative(false);         // false提取索引内点, true提取外点

    		extract.filter(models_points[i]);   // 模型点坐标
		
			extract.setNegative(true);          // true提取拟合模型的外点
			extract.filter(*remain);
			cloud_in.swap(remain); //swap 方法会交换两个智能指针所持有的内部对象的所有权，但不会改变它们指向的对象。

		}
		else
		{
			PCL_ERROR("Could not estimate a line model for the given dataset.\n");
			break;
		}
		
	}

    



}



int main(int argc, char** argv)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_output (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_output_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointXYZRGB point_rgb;
	pcl::io::loadPCDFile("airplane_0002.pcd",*cloud_input);

	std::cout << "Load pcd" << std::endl;

	std::vector<pcl::PointCloud<pcl::PointXYZ>> lines_points;
	uint16_t lines_num = 5;
	lines_points.resize(lines_num);     // 需要 resize 一下，否则 extract.filter(models_points[i]); 报错，如果不 resize，需要用 pushback
	std::vector<pcl::ModelCoefficients> lines_arg; // 模型参数
	

	fitMultiLines(cloud_input, lines_points, lines_arg, lines_num);
	pcl::ExtractIndices<pcl::PointXYZ> extract;

	

	extract.setInputCloud(cloud_input);

	std::vector<uint16_t> r;
	r.push_back(0);
	r.push_back(0);
	r.push_back(255);
	r.push_back(255);
	r.push_back(0);
	std::vector<uint16_t> g;
	g.push_back(0);
	g.push_back(255);
	g.push_back(0);
	g.push_back(255);
	g.push_back(255);
	std::vector<uint16_t> b;
	b.push_back(255);
	b.push_back(0);
	b.push_back(0);
	b.push_back(0);
	b.push_back(255);

	

	for(int i = 0; i<lines_num; i++)
	{
		cloud_output -> clear();
		

		int size = lines_points[i].size();
		// std::cout << size << std::endl;

		for(int k = 0; k < size; k++)
		{
			point_rgb.x = lines_points[i].points[k].x;
			point_rgb.y = lines_points[i].points[k].y;
			point_rgb.z = lines_points[i].points[k].z;
			point_rgb.r = r[i];
        	point_rgb.g = g[i];
        	point_rgb.b = b[i];
			cloud_output_rgb -> push_back(point_rgb);
		}
	}
	
	std::cout << cloud_output_rgb->size() << std::endl;

	cloud_viewer(cloud_output_rgb, 0);

	return 0;
}