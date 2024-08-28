/****** txt 格式点云数据转换为 pcd 格式 (pcl::PointNormal) ******/
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

int main(int argc, char** argv)
{
    std::string input_file = "airplane_0002.txt";
    // 读取txt文件中的点云数据
    std::ifstream infile(input_file.c_str());
    std::vector<float> data;
    float value;

    std::string line;
    while (std::getline(infile, line))
    {
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            float value = std::stof(token);
            data.push_back(value);
        }
    }
    /*
    while (infile >> value)
    {
        data.push_back(value);
        std::cout << "value: " << value << std::endl;
    }
    */
    infile.close();
    // 将数据转换为点云格式
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>);
    int size = data.size() / 6;
    std::cout << "size of pcd: " << size << std::endl;
    cloud->points.resize(size);
    cloud->width = size;
    cloud->height = 1;
    for (int i = 0; i < size; ++i)
    {
        cloud->points[i].x = data[i * 6];
        cloud->points[i].y = data[i * 6 + 1];
        cloud->points[i].z = data[i * 6 + 2];
        cloud->points[i].normal_x = data[i * 6 + 3];
        cloud->points[i].normal_y = data[i * 6 + 4];
        cloud->points[i].normal_z = data[i * 6 + 5];
    }
    std::cout << "point10: " << cloud->points[9].x << ", " << cloud->points[9].y << ", " << cloud->points[9].z << std::endl;
    // 保存为pcd格式的点云文件
    std::string output_file = "airplane_0002.pcd";
    pcl::io::savePCDFileASCII(output_file, *cloud);
    return 0;
}




