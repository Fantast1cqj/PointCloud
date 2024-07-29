#include "preprocess.h"
Preprocess::Preprocess()
{

}

Preprocess::~Preprocess()
{

}

void Preprocess::avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out, bool down_sample)
{
    int debug = 0;
    int ii=0;
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();
    int plsize = msg->point_num;  // point num

    pl_corn.reserve(plsize);
    pl_surf.reserve(plsize);
    pl_full.resize(plsize);

    for(int i=0; i<N_SCANS; i++)
    {
        pl_buff[i].clear();
        pl_buff[i].reserve(plsize);
    }


    for(uint i=1; i<plsize; i++)
    {
      if((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        pl_full[i].x = msg->points[i].x;

        if(debug)
        {
          if(ii < 100)
          {
            ROS_INFO("x = %lf", msg->points[i].x);
            ii++;
          }
          
        }

        pl_full[i].y = msg->points[i].y;
        pl_full[i].z = msg->points[i].z;
        pl_full[i].intensity = msg->points[i].reflectivity;
        pl_full[i].curvature = msg->points[i].offset_time / float(1000000); //use curvature as time of each laser points

        bool is_new = false;
        if((abs(pl_full[i].x - pl_full[i-1].x) > 1e-7)
            || (abs(pl_full[i].y - pl_full[i-1].y) > 1e-7)
            || (abs(pl_full[i].z - pl_full[i-1].z) > 1e-7))
        {
          pl_buff[msg->points[i].line].push_back(pl_full[i]); // pl_buff[1] 中保存了1线的点
        }
      }
    }
    /****** VoxelGrid 降采样 ******/
    if(down_sample)
    {
      pcl::VoxelGrid<PointType> downSizeFilterSurf;
      PointCloudXYZI::Ptr cloud_input(new PointCloudXYZI);
      PointCloudXYZI::Ptr cloud_filterd(new PointCloudXYZI);
      *cloud_input = pl_full;
      downSizeFilterSurf.setInputCloud(cloud_input);  // 设置输入点云
      downSizeFilterSurf.setLeafSize(leafSize, leafSize, leafSize);
      downSizeFilterSurf.filter(*cloud_filterd);

      *pcl_out = *cloud_filterd;
    }
    else
    {
      *pcl_out = pl_full;
    }

    



}


