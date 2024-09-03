#include <iostream>
#include <algorithm>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/random_sample.h>

class KMeans
{

    public:
        KMeans(int max_iteration, int cluster_num);
        double get_distance(pcl::PointXYZ point, pcl::PointXYZ center);
        void kMeans_process(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_input,
                            std::vector<pcl::PointCloud<pcl::PointXYZ>> &cloud_output);

        ~KMeans();

    private:
        int max_iteration_;
        int cluster_num_;

    
};