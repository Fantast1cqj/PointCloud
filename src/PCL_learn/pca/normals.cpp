# include <iostream>
# include <pcl/io/pcd_io.h>
# include "../src/pcd_viewer.h"
# include <pcl/kdtree/kdtree_flann.h>
# include <pcl/common/pca.h>
# include <Eigen/Dense>
int main(int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normal (new pcl::PointCloud<pcl::PointNormal>);

    /****** 读 pcl::PointNormal 的 pcd 文件 ******/
    if(pcl::io::loadPCDFile<pcl::PointNormal>("bed_0003.pcd", *cloud_normal) == -1)
    {
        std::cout << "fale to load" << std::endl;
        return (-1);
    }
    
    std::cout << cloud_normal->points.size() << std::endl;

    /****** 转换为 pcl::PointXYZRGB 格式 ******/
    for (uint16_t i = 0; i < cloud_normal->points.size(); i++) 
    {   
        pcl::PointXYZRGB point;
        point.x = cloud_normal->points[i].x;
        point.y = cloud_normal->points[i].y;
        point.z = cloud_normal->points[i].z;
        
        cloud_xyz->push_back(point);
    }
    std::cout << cloud_normal->points.size() << std::endl;

    /****** 构建 kdTree ******/
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdTree;
    kdTree.setInputCloud(cloud_xyz);

    u_int8_t K = 10;
    

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normal_compute (new pcl::PointCloud<pcl::PointNormal>);
    cloud_normal_compute->resize(cloud_xyz->size());
    for(uint16_t m(0); m < cloud_xyz->size(); m++)
    {
        cloud_normal_compute -> points[m].x = cloud_xyz -> points[m].x;
        cloud_normal_compute -> points[m].y = cloud_xyz -> points[m].y;
        cloud_normal_compute -> points[m].z = cloud_xyz -> points[m].z;
    }
    

    std::cout << "2" << std::endl;


    for(uint16_t k(0); k < cloud_xyz->size(); k++)
    {
        pcl::PointXYZRGB searchPoint;
        std::vector<int> point_ID(K);           // 存储最近邻 10 个点在 cloud 中的 ID
        std::vector<float> point_distance(K);   // 存储 10 个点的距离
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ (new pcl::PointCloud<pcl::PointXYZRGB>);  // 存储最近的10个点
        pcl::PCA<pcl::PointXYZRGB> pca;

        /****** 寻找 searchPoint 的最近邻 ******/
        searchPoint = cloud_xyz -> points[k];
        kdTree.nearestKSearch (searchPoint, K, point_ID, point_distance);
        for(uint8_t t(0); t < 10; t++)
        {
            cloud_->push_back(cloud_xyz -> points[point_ID[t]]);
        }
        
        pca.setInputCloud(cloud_);
        Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
        cloud_normal_compute->points[k].normal_x = eigen_vectors(0, 2);
        cloud_normal_compute->points[k].normal_y = eigen_vectors(1, 2);
        cloud_normal_compute->points[k].normal_z = eigen_vectors(2, 2);
    }


    std::cout << "source: " << cloud_normal->points[100].normal_x << ' ' << cloud_normal->points[100].normal_y << ' ' <<cloud_normal->points[100].normal_z << std::endl;
    std::cout << "compute: " << cloud_normal_compute->points[100].normal_x <<' ' << cloud_normal_compute->points[100].normal_y << ' ' <<cloud_normal_compute->points[100].normal_z << std::endl;

    std::cout << "source: " << cloud_normal->points[300].normal_x << ' ' << cloud_normal->points[300].normal_y << ' ' <<cloud_normal->points[300].normal_z << std::endl;
    std::cout << "compute: " << cloud_normal_compute->points[300].normal_x <<' ' << cloud_normal_compute->points[300].normal_y << ' ' <<cloud_normal_compute->points[300].normal_z << std::endl;



    cloud_viewer(cloud_normal_compute, 2);
    return 0;
}