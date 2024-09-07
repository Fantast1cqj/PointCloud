/****** 点云 PCA 求主成分方向 ******/
# include <iostream>
# include <fstream>
# include <string>
# include <vector>
# include <Eigen/Dense>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/io/pcd_io.h>
# include "../src/pcd_viewer.h"

int main(int argc, char** argv)
{   
    std::ifstream file("airplane_0001.txt");
    if(!file.is_open())
    {
        std::cerr << "can not open the file" << std::endl;
        return 1;
    }

    uint8_t i(0);
    uint8_t k(0);
    uint8_t m(0);

    std::vector<std::vector<float>> tempData;

    std::string line;
    while (std::getline(file, line))
    {
        // line 为 txt 文件的一行
        /*
        if(m == 0)
        {
            m++;
            std::cout << line << std::endl;
        } */
        std::stringstream ss(line);    // 某行的字符串转为数据流 ss
        std::vector<float> row_data;
        float value;
        std::string token;
        while(std::getline(ss, token,','))   // 从输入流 ss 中读取字符，直到遇到指定的分隔符，将读取的内容存储到 token
        {
            /*
            if(i < 3)
            {
                i++;
                // std::cout << token << std::endl;
            }   */
            std::stringstream(token) >> value;  // token 转为 float
            row_data.push_back(value);
        }
        /*
        if(k<1)
        {
            k++;
            std::cout << row_data[0] << ", " << row_data[1]<< ", "<< row_data[2] << ", " << row_data[3] << std::endl;
        }  */
        tempData.push_back(row_data);
    }

    // std::cout << tempData.size() << std::endl;    tempData 的行数
    file.close();
    Eigen::MatrixXf data(tempData.size(), 3);
    for(int t(0); t<tempData.size(); t++)
    {
        for(int j(0);j<3;j++)
        {
            data(t,j) = tempData[t][j];
        }
    }
    // std::cout << data << std::endl;
    std::cout << "size of data: " <<data.size() << std::endl;

    /****** 归一化 ******/
    Eigen::Vector3f centroid = data.colwise().mean();   // 计算每一列的平均值
    data.rowwise() -= centroid.transpose();             // 每行数据减去平均值
    // std::cout << data << std::endl;
    centroid = data.colwise().mean();
    std::cout << "mean: \n" << centroid << std::endl;

    /****** 协方差矩阵 ******/
    Eigen::MatrixXf covariance_matrix = (data.transpose() * data) / data.rows();
    std::cout << "协方差矩阵：" << std::endl;
    std::cout << covariance_matrix << std::endl;

    /******* 特征值和特征向量 ******/
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix);
    Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();    // col(0) 对应特征值最小
    Eigen::Vector3f eigen_values = eigen_solver.eigenvalues();      // 特征值从小到大

    std::cout << "Eigen Vectors: " << std::endl << eigen_vectors << std::endl;
    std::cout << "Eigen Values: " << std::endl << eigen_values << std::endl;

    /*
    Eigen::MatrixXf data_test(5, 3);
    for(i = 0;i<3;i++)
    {
        data_test(0,i) = 1;
        data_test(1,i) = 2;
        data_test(2,i) = 3;
        data_test(3,i) = 4;
        data_test(4,i) = 8;

    }
    std::cout<<"data_test: "<< '\n' <<data_test<<std::endl;
    Eigen::Vector3f centroid_test = data_test.colwise().mean();
    std::cout << "mean: \n" << centroid_test.transpose() << std::endl;    */

    /****** 加载点云 ******/
    pcl::PCLPointCloud2 cloud_binary2;
    pcl::PointCloud<pcl::PointNormal>::Ptr airplane_0001 (new pcl::PointCloud<pcl::PointNormal>);
    if( pcl::io::loadPCDFile("airplane_0001.pcd", cloud_binary2) == -1)
    {
        std::cout << "找不到 pcd 文件" << std::endl;
        return -1;
    }    
    pcl::fromPCLPointCloud2(cloud_binary2, *airplane_0001);

    /****** 点云可视化 ******/
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer("Viewer1"));
    viewer->addPointCloudNormals<pcl::PointNormal>(airplane_0001, 1, 0.1, "normals");
    viewer -> setBackgroundColor(0, 0, 0);
    Eigen::Vector3f start(0.0, 0.0, 0.0);
    Eigen::Vector3f end1(eigen_vectors.col(0));
    Eigen::Vector3f end2(eigen_vectors.col(1));
    Eigen::Vector3f end3(eigen_vectors.col(2));

    // std::cout << "Vector 1: " << end1.transpose() << std::endl;
    // std::cout << "Vector 2: " << end2.transpose() << std::endl;
    // std::cout << "Vector 3: " << end3.transpose() << std::endl;
    

    pcl::PointXYZ pcl_start(start[0], start[1], start[2]);
    pcl::PointXYZ pcl_end1(end1[0], end1[1], end1[2]);
    pcl::PointXYZ pcl_end2(end2[0], end2[1], end2[2]);
    pcl::PointXYZ pcl_end3(end3[0], end3[1], end3[2]);

    viewer->addArrow(pcl_end1, pcl_start, 1.0, 0, 0, false, "arrow1");
    viewer->addArrow(pcl_end2, pcl_start, 1.0, 0, 0, false, "arrow2");
    viewer->addArrow(pcl_end3, pcl_start, 1.0, 0, 0, false, "arrow3");
    


    pcl::PCA<pcl::PointNormal> pca;
    pca.setInputCloud(airplane_0001);
    Eigen::Vector3f eigen_values2 = pca.getEigenValues();
    Eigen::Matrix3f eigen_vectors2 = pca.getEigenVectors();

    std::cout << "Eigen Values(PCL): \n" << eigen_values2 << std::endl;
    std::cout << "Eigen Vectors(PCL): \n" << eigen_vectors2 << std::endl;


    while (!viewer->wasStopped())
    {
        viewer->spinOnce(10);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));   // 窗口 10 ms 刷新一次
    }





    return 0;
}