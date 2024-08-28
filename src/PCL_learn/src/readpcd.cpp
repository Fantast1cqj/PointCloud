/**************** 读取 pcd 文件 ****************/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>


int main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    // 创建智能指针 cloud， new 一个对象


    // 由于点云的动态特性，更喜欢读取为二进制 blob，然后转换为我们想要使用的实际表示形式。
    pcl::PointCloud<pcl::PointXYZ> cloud2;
    pcl::PCLPointCloud2 cloud_blob;                       // 点云的二进制形式
    pcl::io::loadPCDFile("test_pcd_1.pcd", cloud_blob);   // 加载点云
    pcl::fromPCLPointCloud2(cloud_blob, cloud2);          // 转换点云形式


    if(pcl::io::loadPCDFile<pcl::PointXYZ>("test_pcd_1.pcd", *cloud) == -1)
    // 加载 PCD 文件到 cloud 指向的点云对象中
    {
        std::cout << "fale to load" << std::endl;
        return (-1);
    }

    for(const auto &point: *cloud)
    {
        std::cout << "point x: " << point.x << " y: " << point.y << " z: " << point.z << std::endl; 
    }


    
  return (0);
}