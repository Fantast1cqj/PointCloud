#include <ros/ros.h>
#include <livox_ros_driver/CustomMsg.h>
#include <Eigen/Core>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include "preprocess.h"

shared_ptr<Preprocess> p_pre(new Preprocess());
std::deque<double>                     time_buffer;
std::deque<PointCloudXYZI::Ptr>        lidar_buffer;
PointCloudXYZI::Ptr cloud2pcd(new PointCloudXYZI);

double last_timestamp_lidar(0);
size_t lidar_buffer_size(0);
std::string filename = "output.pcd";
long int cloud2pcd_width(0);


void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    bool debug(0);
    // mtx_buffer.lock();
    // double preprocess_start_time = omp_get_wtime();
    // scan_count ++;
    // lidar_buffer.pop_back();

    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    // if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    // {
    //     printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    // }

    // if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    // {
    //     timediff_set_flg = true;
    //     timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
    //     printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    // }

    // wmywmy transform point into virtual lidar frame
    Eigen::Vector3d temp1, temp2;
    livox_ros_driver::CustomMsg::Ptr new_msg(new livox_ros_driver::CustomMsg(*msg));
    for(int i=0; i<new_msg->points.size(); i++)
    {
        temp1[0] = new_msg->points[i].x;
        temp1[1] = new_msg->points[i].y;
        temp1[2] = new_msg->points[i].z;
        // temp2 = virtualLidar_R * temp1;   // temp1 进行旋转
        // new_msg->points[i].x = temp2[0];
        // new_msg->points[i].y = temp2[1];
        // new_msg->points[i].z = temp2[2];
    }
    // wmywmy
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->avia_handler(new_msg, ptr, 0);

    if(debug)
    {   
        ROS_INFO("x = %lf", ptr->points[2000].x);
    }

    lidar_buffer.push_back(ptr);     // 在 lidar_buffer 最后面添加点云
    // lidar_buffer.push_front(ptr);    // 在 lidar_buffer 最前面添加点云
    time_buffer.push_back(last_timestamp_lidar);
    
    // s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;  // 查看处理时间
    // mtx_buffer.unlock();
    // sig_buffer.notify_all();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "motion_comp_node");
    ros::NodeHandle nh;

    ros::Rate rate = 100;

    ros::Subscriber sub_pcl = 
        nh.subscribe("/livox/lidar", 200000, livox_pcl_cbk, ros::TransportHints().tcpNoDelay());

    while(ros::ok())
    {
        
        lidar_buffer_size = lidar_buffer.size();

        /****** 保存 lidar_buffer 中前 50 个点云到 cloud2pcd ******/
        if(lidar_buffer_size == 50)
        {
            cloud2pcd->height = 1;
            cloud2pcd->width = 0;

            /*test */
            // cloud2pcd_width = lidar_buffer[20]->size();
            // ROS_INFO("size %d", cloud2pcd_width);
            // cloud2pcd->width = cloud2pcd_width;
            // auto point_ptr = lidar_buffer[20];
            // cloud2pcd->points.resize(cloud2pcd->width * cloud2pcd->height);
            // pcl::io::savePCDFileASCII(filename, *cloud2pcd);

            for(uint i = 0; i<lidar_buffer_size; i++)
            {
                /****** 方式 1 ******/
                // cloud2pcd_width = cloud2pcd_width + (lidar_buffer[i]->size());   // cloud2pcd_width 必须为long int
                // cloud2pcd->width = cloud2pcd_width;
                // cloud2pcd->points.resize(cloud2pcd->width * cloud2pcd->height);
                // *cloud2pcd = *cloud2pcd + *lidar_buffer[i];
                
                /****** 方式 2 ******/
                cloud2pcd->points.insert(cloud2pcd->points.end(), lidar_buffer[i]->points.begin(), lidar_buffer[i]->points.end());   // 有问题
            }
            cloud2pcd->width = cloud2pcd->points.size();  // 别忘了更新 cloud2pcd->width ！！！
            pcl::io::savePCDFileASCII(filename, *cloud2pcd);
            ROS_INFO("size %ld", cloud2pcd_width);
            ROS_INFO("50 frame save sucess");
            lidar_buffer.clear();
            cloud2pcd->clear();
        }
      


        ros::spinOnce();
        rate.sleep();
    }



}