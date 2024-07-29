#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver/CustomMsg.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

class Preprocess
{
    public:
    Preprocess();
    ~Preprocess();

    int N_SCANS = 6;  // 雷达扫描线数？
    PointCloudXYZI pl_buff[128]; //maximum 128 line lidar

    // void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out, bool down_sample);


    private:
    PointCloudXYZI pl_full, pl_corn, pl_surf, pl_down;
    float leafSize = 0.01f; // 降采样体素网格大小
};