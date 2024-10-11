#include <iostream>
#include <pcl/io/pcd_io.h>                    
#include <pcl/point_types.h>                         
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>



void fitMultiLines(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::vector<pcl::PointIndices>& models_index,std::vector<pcl::ModelCoefficients>& models_arg, uint16_t line_num)
{
	pcl::PointIndices::Ptr Indexes(new pcl::PointIndices);  // 拟合点云的索引

    pcl::SACSegmentation<pcl::PointXYZ> seg; // 分割器
	seg.setOptimizeCoefficients(true);       // 可选择配置，设置模RANSAC型系数需要优化
    seg.setModelType(pcl::SACMODEL_LINE);    // 拟合目标形状 line 
    seg.setMethodType(pcl::SAC_RANSAC);      // 拟合方法：随机采样法
    seg.setDistanceThreshold(0.05);          // 设置误差容忍范围，也就是阈值，直线模型的 宽度
    seg.setMaxIterations(500);               // 最大迭代次数，默认迭代50次

	
	for(uint16_t i(0); i < line_num; i++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr remain (new pcl::PointCloud<pcl::PointXYZ>);
		pcl::ModelCoefficients::Ptr arg;   // 拟合模型的参数
		seg.setInputCloud(cloud_in);  	// input
		seg.segment(*Indexes, *arg);  	// 拟合点云
		if(Indexes -> indices.size() > 20)
		{
			models_index.push_back(*Indexes);
			models_arg.push_back(*arg);
			
			pcl::ExtractIndices<pcl::PointXYZ> extract; // 索引提取器
			extract.setInputCloud(cloud_in);    // 设置输入点云
			extract.setIndices(Indexes);        // 设置索引
			extract.setNegative(true);          // true提取拟合模型的外点
			extract.filter(*remain);

			cloud_in.swap(remain);
		}
		else
		{
			PCL_ERROR("Could not estimate a line model for the given dataset.\n");
			break;
		}
		
	}

    



}



int mait(int argc, char** argv)
{
	return 0;
}



void fitMultipleLines(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<pcl::ModelCoefficients>& lineCoff)
{
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	pcl::SACSegmentation<pcl::PointXYZ> seg;               // 创建拟合对象
	seg.setOptimizeCoefficients(true);                     // 设置对估计模型参数进行优化处理
	seg.setModelType(pcl::SACMODEL_LINE);                  // 设置拟合模型为直线模型
	seg.setMethodType(pcl::SAC_RANSAC);                    // 设置拟合方法为RANSAC
	seg.setMaxIterations(1000);                             // 设置最大迭代次数
	seg.setDistanceThreshold(0.1);                       // 判断是否为模型内点的距离阀值/设置误差容忍范围

	int i = 0, nr_points = cloud->points.size();
	int k = 0;
	while (k < 5 && cloud->points.size() > 0.1 * nr_points)// 从0循环到5执行6次，并且每次cloud的点数必须要大于原始总点数的0.1倍
	{
		pcl::ModelCoefficients coefficients;
		seg.setInputCloud(cloud);                         // 输入点云						 
		seg.segment(*inliers, coefficients);              // 内点的索引，模型系数

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_line(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr outside(new pcl::PointCloud<pcl::PointXYZ>);
		if (inliers->indices.size() > 20) // 判断提取直线上的点数是否小于20个点，小于的话该直线不合格
		{
			lineCoff.push_back(coefficients);             // 将参数保存进vector中
			pcl::ExtractIndices<pcl::PointXYZ> extract;   // 创建点云提取对象
			extract.setInputCloud(cloud);
			extract.setIndices(inliers);
			extract.setNegative(false);                   // 设置为false，表示提取内点
			extract.filter(*cloud_line);

			extract.setNegative(true);                    // true提取外点（该直线之外的点）
			extract.filter(*outside);                     // outside为外点点云
			cloud.swap(outside);                          // 将cloud_f中的点云赋值给cloud
		}
		else
		{
			PCL_ERROR("Could not estimate a line model for the given dataset.\n");
			break;
		}

		std::stringstream ss;
		ss << "line_" << i + 1 << ".pcd"; // 记录提取的是第几条直线，并以该序号命名输出点云
		pcl::PCDWriter writer;
		writer.write<pcl::PointXYZ>(ss.str(), *cloud_line, false);

		i++;
		k++;
	}
}

int main(int argc, char** argv)
{
	// 加载点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("data//L.pcd", *cloud) == -1)
	{
		PCL_ERROR("点云读取失败 \n");
		return (-1);
	}

	vector<pcl::ModelCoefficients> LinesCoefficients;
	fitMultipleLines(cloud, LinesCoefficients);

	cout << "一共拟合出" << LinesCoefficients.size() << "条直线，直线系数分别为：\n" << endl;

	for (auto l : LinesCoefficients)
	{
		cout << l.values[0] << "," << l.values[1]
			<< "," << l.values[2] << "," << l.values[3]
			<< "," << l.values[4] << "," << l.values[5] << endl;
	}

	return 0;
}

