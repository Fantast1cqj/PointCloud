#include "kmeans.h"
int main(int argc, char** argv)
{
    KMeans test(3, 5); // 聚类个数：5
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::PointCloud<pcl::PointXYZ>> output_cluster_cloud;
    pcl::io::loadPCDFile("airplane_0001.pcd",*cloud);

    // std::cout<< "load file ok"<<std::endl;

    test.kMeans_process(cloud, output_cluster_cloud);
    std::cout<< "kMeans_process ok"<<std::endl;

    

    std::vector<double> distance = {0.277537, 0.652093, 1.03941, 0.471525, 0.227368};
    auto min_it = std::min_element(distance.begin(), distance.end());
	size_t min_index = std::distance(distance.begin(), min_it);
    std::cout<< "min pos" << min_index << std::endl;


    for (int i = 0; i < 5; ++i)
    {
        
        std::cerr <<"output_cloud[i].points.size()"<< output_cluster_cloud[i].points.size()<<std::endl;
        output_cluster_cloud[i].width = output_cluster_cloud[i].points.size();
        output_cluster_cloud[i].height = 1;
        output_cluster_cloud[i].resize(output_cluster_cloud[i].width * output_cluster_cloud[i].height);
        pcl::io::savePCDFile( "kmeans"+std::to_string(i) + ".pcd", output_cluster_cloud[i]);
        //pcl::io::savePCDFile()
    }

}