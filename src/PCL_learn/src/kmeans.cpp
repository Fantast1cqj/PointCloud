#include "kmeans.h"
KMeans::KMeans(int max_iteration, int cluster_num)
{
    max_iteration_ = max_iteration;
    cluster_num_ = cluster_num;
}

KMeans::~KMeans()
{

}

void KMeans::kMeans_process(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_input,
                    std::vector<pcl::PointCloud<pcl::PointXYZ>> &cloud_output)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    /****** 初始化中心点 ******/
    pcl::RandomSample<pcl::PointXYZ> random_sample;
    random_sample.setInputCloud(cloud_input);
    random_sample.setSample(cluster_num_);
    random_sample.filter(*center_cloud);

    int iterations = 0;
    while(iterations < max_iteration_)
    {



        ++max_iteration_;
    }




                                
}





void KMeans::kMeans(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<pcl::PointCloud<pcl::PointXYZ>> &cluster_cloud1)
{
	
	if (!cloud->empty()&&!centre_points_->empty())
	{
		unsigned int iterations = 0;
		double sum_diff = 0.2;
		std::vector<pcl::PointCloud<pcl::PointXYZ>>cluster_cloud;
		while (!(iterations>=max_iteration_||sum_diff<=0.05))//如果大于迭代次数或者两次重心之差小于0.05就停止
		//while ((iterations<= max_iteration_ || sum_diff >= 0.1))
	
		{
			sum_diff = 0;
			std::vector<int> points_processed(cloud->points.size(), 0);
			cluster_cloud.clear();
			cluster_cloud.resize(cluster_num_);
			for (size_t i = 0; i < cloud->points.size(); ++i)
			{
				if (!points_processed[i])
				{
					std::vector<double>dists(0, 0);
					for (size_t j = 0; j < cluster_num_; ++j)
					{						
						dists.emplace_back(pointsDist(cloud->points[i], centre_points_->points[j]));
					}
					std::vector<double>::const_iterator min_dist = std::min_element(dists.cbegin(), dists.cend());
					unsigned int it = std::distance(dists.cbegin(), min_dist);//获取最小值所在的序号或者位置（从0开始）
					//unsigned int it=std::distance(std::cbegin(dists), min_dist);
					cluster_cloud[it].points.push_back(cloud->points[i]);//放进最小距离所在的簇
					points_processed[i] = 1;
				}

				else
					continue;

			}
			//重新计算簇重心
			pcl::PointCloud<pcl::PointXYZ> new_centre;
			for (size_t k = 0; k < cluster_num_; ++k)
			{
				Eigen::Vector4f centroid;
				pcl::PointXYZ centre;
				pcl::compute3DCentroid(cluster_cloud[k], centroid);
				centre.x = centroid[0];
				centre.y = centroid[1];
				centre.z = centroid[2];
				//centre_points_->clear();
				//centre_points_->points.push_back(centre);
				new_centre.points.push_back(centre);

			}
			//计算重心变化量
			for (size_t s = 0; s < cluster_num_; ++s)
			{
				std::cerr << " centre" << centre_points_->points[s] << std::endl;

				std::cerr << "new centre" << new_centre.points[s] << std::endl;
				sum_diff += pointsDist(new_centre.points[s], centre_points_->points[s]);

			}
			std::cerr << sum_diff << std::endl;
			centre_points_->points.clear();
			*centre_points_ = new_centre;
			
			++iterations;
		}
		std::cerr << cluster_cloud[0].size() << std::endl;
		std::cerr << cluster_cloud[1].size() << std::endl;

		cluster_cloud1.assign(cluster_cloud.cbegin(),cluster_cloud.cend());//复制点云向量

	}

}
