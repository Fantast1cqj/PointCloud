/**************** 点云可视化 ****************/
# include "pcd_viewer.h"


using namespace std::this_thread;
using namespace std::chrono;
// 简单的可视化函数
void cloud_viewer_simple (pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    // CloudViewer 类不适合在多线程应用程序中使用！
    pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
    viewer.showCloud (cloud);
    while (!viewer.wasStopped ())
    {

    }
}

void cloud_viewer(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, u_int8_t mod)
{
    // 处理 pcl::PointXYZ 类型的点云
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer("Viewer"));
    // viewer->addPointCloudNormals<pcl::PointNormal>(cloud, 1, 0.1, "normals");
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");  // 加个坐标系
    viewer->addCoordinateSystem(1.0);
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(10);
        std::this_thread::sleep_for(10ms);
    }
}


void cloud_viewer(pcl::PointCloud<pcl::PointNormal>::ConstPtr cloud, u_int8_t mod)
{
    // 处理 pcl::PointNormal 类型的点云
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer("Viewer"));
    viewer->addPointCloudNormals<pcl::PointNormal>(cloud, 1, 0.1, "normals");

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(10);
        std::this_thread::sleep_for(10ms);
    }
}

void cloud_viewer(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, u_int8_t mod)
{
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer("Viewer"));
    if(mod == 0)
        {
        // 简单可视化 (XYZRGB)
        viewer->setBackgroundColor(255, 255, 255);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
        viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "milk");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "milk");
        viewer->addCoordinateSystem(0.1);
        viewer->initCameraParameters();
        }
        else if(mod == 1)
        {
            // 设置点云颜色颜色
            viewer -> setBackgroundColor(0, 0, 0);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(cloud, 0, 255, 0);  // 设置点云颜色
            viewer -> addPointCloud<pcl::PointXYZRGB> (cloud, single_color, "sample cloud");
            viewer -> setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
            viewer -> addCoordinateSystem (0.1);
            viewer -> initCameraParameters ();
        }
        while (!viewer->wasStopped())
        {
            viewer->spinOnce(10);
            std::this_thread::sleep_for(10ms);
        }
}




void cloud_viewer (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals,u_int8_t mod)
{
    //---------------------------------------
    //----------- normals 点云法线 -----------
    //---------------------------------------
    if(mod == 2)
    {
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("Viewer"));

        viewer -> setBackgroundColor(0, 0, 0);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
        viewer -> addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
        viewer -> setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");

        viewer -> addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals, 10, 0.05, "normals");    // 设置点云法线

        viewer -> addCoordinateSystem (1.0);
        viewer -> initCameraParameters ();

        while (!viewer -> wasStopped ())
        {
            viewer -> spinOnce (10);             // 响应用户的交互操作
            std::this_thread::sleep_for(10ms);   // 窗口 10 ms 刷新一次
        }
    }
    else
    {
        std::cout << "cloud viewer mod error !!!" << std::endl;
        PCL_ERROR ("cloud viewer mod error !!! \n");
    }
}




// int main(int argc, char** argv)
// {

//     pcl::PCLPointCloud2 cloud_binary;
//     if( pcl::io::loadPCDFile("airplane_0001.pcd", cloud_binary) == -1)
//     {
//         std::cout << "找不到 pcd 文件" << std::endl;
//         return -1;
//     }
//     pcl::PointCloud<pcl::PointNormal>::Ptr cloud_airplane_0001 (new pcl::PointCloud<pcl::PointNormal>);
   

//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_airplane_0001_1 (new pcl::PointCloud<pcl::PointXYZRGB>);


//     pcl::fromPCLPointCloud2(cloud_binary, *cloud_airplane_0001_1);
    
//     cloud_viewer(cloud_airplane_0001_1, 1);
//     // cloud_viewer_simple(cloud_airplane_0001_1);
//     // cloud_viewer(cloud_airplane_0001, 1);


//     return 0;
// }