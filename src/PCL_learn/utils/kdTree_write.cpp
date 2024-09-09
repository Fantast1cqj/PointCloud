/**************** 使用 Kd Tree 搜索点云 ****************/

# include <pcl/point_cloud.h>
# include <pcl/kdtree/kdtree_flann.h>

# include <iostream>
# include <vector>
# include <ctime>

int main(int argc, char** argv)
{
    srand(time(NULL));   // 随机数生成 time(NULL)返回当前秒数，程序每次运行随机数不同
    // std::cout << time(NULL) << std::endl;

    // 定义点云指针
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    cloud -> width = 1000;
    cloud -> height = 1;
    (*cloud).points.resize((*cloud).width * (*cloud).height);
    
    // 设置初始点云 0-1024 之间
    for(u_int16_t i = 0;i < cloud->size(); ++i)
    {   
        (*cloud)[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
        (*cloud)[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
        (*cloud)[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);

    }

    // 定义 kdTree
    pcl::KdTreeFLANN<pcl::PointXYZ> kdTree;
    kdTree.setInputCloud(cloud);

    // 定义搜索点
    pcl::PointXYZ searchPoint;
    searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
    searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
    searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);
    

    // 最近邻搜索
    u_int8_t K = 10;
    std::vector<int> point_ID(K);           // 存储最近邻 10 个点在 cloud 中的 ID
    std::vector<float> point_distance(K);   // 存储 10 个点的距离
    // kdTree.nearestKSearch (searchPoint, K, point_ID, point_distance);
    if(kdTree.nearestKSearch (searchPoint, K, point_ID, point_distance) > 0)
    {
        for (std::size_t i = 0; i < point_ID.size (); ++i)
        {
            std::cout << "    "  <<   (*cloud)[ point_ID[i] ].x 
            << " " << (*cloud)[ point_ID[i] ].y 
            << " " << (*cloud)[ point_ID[i] ].z 
            << " (squared distance: " << point_distance[i] << ")" << std::endl;
        }
    }

    // 范围搜索
    // float radius = 256.0f * rand () / (RAND_MAX + 1.0f);
    float radius = 200;                  // 搜索范围
    std::vector<int> point_ID2;
    std::vector<float> point_distance2;  // 距离的平方


    std::cout << "Neighbors within radius search at (" << searchPoint.x 
            << " " << searchPoint.y 
            << " " << searchPoint.z
            << ") with radius=" << radius << std::endl;

    if ( kdTree.radiusSearch (searchPoint, radius, point_ID2, point_distance2) > 0 )  // point_distance2 存储距离的平方
    {
        for (std::size_t i = 0; i < point_ID2.size (); ++i)
        {
            std::cout << "    "  <<   (*cloud)[ point_ID2[i] ].x 
                << " " << (*cloud)[ point_ID2[i] ].y 
                << " " << (*cloud)[ point_ID2[i] ].z 
                << " (squared distance: " << point_distance2[i] << ")" << std::endl;
        }
    }


    return 0;
}
