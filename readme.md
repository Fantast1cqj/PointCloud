**<font size=6>Point Cloud</font>**
<!-- <link rel="stylesheet" type="text/css" href="auto-number-title.css" /> -->

Markdown 教程：https://markdown.com.cn/basic-syntax/
- [1. PCL](#1-pcl)
  - [1.1 Filter](#11-filter)
    - [pass through 直通滤波器](#pass-through-直通滤波器)
    - [voxel grid 体素滤波器](#voxel-grid-体素滤波器)
    - [indices 索引提取](#indices-索引提取)
    - [denoise 去除噪声](#denoise-去除噪声)
  - [1.2 RANSAC (Random sample consensus)](#12-ransac-random-sample-consensus)
    - [拟合直线](#拟合直线)
    - [多直线拟合](#多直线拟合)
    - [平面拟合](#平面拟合)
  - [点云配准](#点云配准)
    - [4PCS(4-Point Congruent Sets) 点云粗配准](#4pcs4-point-congruent-sets-点云粗配准)
    - [K4PCS(4-Point Congruent Sets) 点云粗配准](#k4pcs4-point-congruent-sets-点云粗配准)
    - [RANSAC 点云粗配准](#ransac-点云粗配准)
    - [点到点 ICP 点云精配准](#点到点-icp-点云精配准)
    - [kdtree 优化 ICP](#kdtree-优化-icp)
  - [点云分割](#点云分割)
    - [欧式聚类分割](#欧式聚类分割)
    - [区域生长分割](#区域生长分割)
  - [1.5 特征点与特征描述](#15-特征点与特征描述)
    - [1.5.1 pcl::Feature](#151-pclfeature)
    - [1.5.2 点云 PCA](#152-点云-pca)
    - [1.5.3 求点云曲率](#153-求点云曲率)
    - [1.5.4 MLS 平滑点云并计算法向量](#154-mls-平滑点云并计算法向量)
    - [1.5.5 计算点云包围盒](#155-计算点云包围盒)
    - [1.5.6 点云边界提取](#156-点云边界提取)
    - [1.5.7  Alpha Shapes 平面点云边界特征提取](#157--alpha-shapes-平面点云边界特征提取)
- [2. Lidar 运动补偿](#2-lidar-运动补偿)
  - [点云预处理](#点云预处理)
  - [使用 IMU 进行补偿](#使用-imu-进行补偿)
    - [时间戳问题](#时间戳问题)
- [3. 点云配准](#3-点云配准)
- [4. 深度学习基础](#4-深度学习基础)
  - [4.1 多层感知机 (MLP)](#41-多层感知机-mlp)
    - [激活函数](#激活函数)
    - [训练误差与泛化误差](#训练误差与泛化误差)
    - [训练集、验证集、测试集](#训练集验证集测试集)
    - [欠拟合与过拟合](#欠拟合与过拟合)
    - [权重衰减](#权重衰减)
    - [前向传播、反向传播、计算图](#前向传播反向传播计算图)
    - [参数初始化](#参数初始化)
  - [4.2 深度学习计算](#42-深度学习计算)
    - [层和块](#层和块)
    - [参数管理、读写文件](#参数管理读写文件)
  - [4.3 卷积神经网络](#43-卷积神经网络)
    - [填充和步幅](#填充和步幅)
    - [多通道输入输出](#多通道输入输出)
  - [4.4 Transformer](#44-transformer)
  - [4.5 注意力机制](#45-注意力机制)
  - [4.6 nn.Conv1d](#46-nnconv1d)
  - [4.7 nn.Conv2d](#47-nnconv2d)
  - [4.8 nn.BatchNorm1d](#48-nnbatchnorm1d)
  - [4.9 nn.LayerNorm](#49-nnlayernorm)
- [5. 点云语义分割](#5-点云语义分割)
  - [5.1 Point Net](#51-point-net)
  - [5.2 Point Net++](#52-point-net)
  - [5.3 RangeNet++](#53-rangenet)
- [6. 点云补全](#6-点云补全)
  - [partial to complete](#partial-to-complete)
  - [配 EditVAE 环境](#配-editvae-环境)
  - [AnchorFormer](#anchorformer)
    - [Anchor Generation](#anchor-generation)
    - [Anchor Scattering](#anchor-scattering)
    - [Point Morphing](#point-morphing)
  - [Point Transformer](#point-transformer)
    - [Point Transformer Layer](#point-transformer-layer)
    - [Position Encoding](#position-encoding)
    - [Point Transformer Block](#point-transformer-block)
    - [Network Architecture](#network-architecture)
  - [SeedFormer](#seedformer)
    - [Architecture Overview](#architecture-overview)
    - [Point Cloud Completion with Patch Seeds](#point-cloud-completion-with-patch-seeds)
    - [确定一个 seedformer 测试集](#确定一个-seedformer-测试集)
  - [PointAttN](#pointattn)
  - [WalkFormer](#walkformer)
    - [Point Walk](#point-walk)
  - [Zero-shot](#zero-shot)
    - [Point Cloud Colorization:](#point-cloud-colorization)
  - [CDPNet (AAAI 2024)](#cdpnet-aaai-2024)
    - [Introduction:](#introduction)
- [7. Conda](#7-conda)
  - [解决 conda 权限问题](#解决-conda-权限问题)
- [8. 实验记录](#8-实验记录)
  - [model\_addkp](#model_addkp)
    - [v 0.5](#v-05)



# 1. PCL

## 1.1 Filter

### pass through 直通滤波器
code: [pass_through.cpp](src/PCL_learn/filter/pass_through.cpp)

PCL 直通滤波器 pcl::PassThrough\<pcl::PointXYZ\> filter, 对坐标某一范围内进行去除或保留

### voxel grid 体素滤波器
code: [voxel_grid.cpp](src/PCL_learn/filter/voxel_grid.cpp)

1. 创建 voxel gird 进行下采样，用体素 **重心** 近似体素内的其他点，比体素中心更慢，但是表示曲面更准确
2. Approximate Voxel Grid （使用体素中心）
3. 改进 Voxel Grid，使用原始点云距离重心最近的点作为下采样的点

###  indices 索引提取
code: [indices.cpp](src/PCL_learn/filter/indices.cpp)

根据点云索引对点进行提取

    pcl::PointIndices indices;
    uint16_t i = 0;
    for(i = 0; i < 50; i++)
    {
        indices.indices.push_back(i);
    }

    pcl::ExtractIndices<pcl::PointXYZ> extr; // 索引提取器
    extr.setInputCloud(cloud_input);           // 设置输入点云
    extr.setIndices(boost::make_shared<const pcl::PointIndices>(indices)); // 设置索引 创建一个共享智能指针
    extr.filter(*cloud_output);     // 提取出 indices 中的点云

### denoise 去除噪声

code: [denoise.cpp](src/PCL_learn/filter/denoise.cpp)

1. 半径滤波：半径 r，点数 n，遍历每个点，点为球心，半径 r，球内点数少于 n，去除该点
2. 统计滤波：遍历所有点，取某个点周围 k 个点，算 k 个距离，并计算距离的均值和方差，保留 (μ - std * σ, μ + std * σ) 距离内的点
3. Gaussian 滤波：原理与图像高斯滤波相似，点坐标为周围点坐标的高斯加权

## 1.2 RANSAC (Random sample consensus)
RANSAC 用于从包含大量异常值（噪声或离群点）的数据中拟合数学模型。




普通最小二乘：在现有数据下，如何实现最优。是从一个整体误差最小的角度去考虑，尽量谁也不得罪。容易受到噪点影响

RANSAC：首先假设数据具有某种特性（目的），为了达到目的，适当割舍一些现有的数据。

**思路：**
1. 设定要拟合的模型，随机抽取样本点，拟合模型
2. 找到拟合模型容忍误差范围内的点个数
3. 重新选取样本点，重复1、2迭代
4. 选取误差范围内点最多的一次拟合


**RANSAC算法的期望概率（Probability）** ：算法至少找到一组纯内点样本的概率​，通常设为0.99，与迭代次数相关，值越大迭代次数越高
PCL默认根据概率公式自动计算迭代次数，手动设置setMaxIterations会覆盖此逻辑



**应用：**

1. 拟合空间直线：采样点为 2，6个参数
2. 拟合圆柱：圆柱轴上一点的坐标（x,y,z）、圆柱轴方向向量（x,y,z）、圆柱半径 r 。共7个参数
3. 拟合平面：平面归一化法向量（x,y,z），平面位置（平面沿法向量方向到原点的有符号距离）。共4个参数
4. 拟合球：球心和半径。共4个参数




### 拟合直线
code: [line.cpp.cpp](src/PCL_learn/RANSAC/line.cpp)

RANSAC 提取直线

RANSAC 提取有角度约束的直线 设置坐标轴和坐标轴的夹角

最小二乘缺陷：全局最优解，有的数据是噪声，不适合求解

    pcl::SACSegmentation<pcl::PointXYZ> seg;


### 多直线拟合

code: [multi_line.cpp](src/PCL_learn/RANSAC/multi_line.cpp)

使用 RANSAC 拟合多条直线

    cloud_in.swap(remain); //swap 方法会交换两个智能指针所持有的内部对象的所有权，但不会改变它们指向的对象。

### 平面拟合

code: [plane.cpp](src/PCL_learn/RANSAC/plane.cpp)

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr remain_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);   // 智能指针在赋值的时候需要 resize，pushback()不需要   变量赋值不需要 resize
    remain_rgb -> width  = remain_size;
    remain_rgb -> height = 1;
    remain_rgb -> resize(remain_size);
    {
      remain_rgb -> points[k].x = remain -> points[k].x;
      remain_rgb -> points[k].y = remain -> points[k].y;
      remain_rgb -> points[k].z = remain -> points[k].z;
      remain_rgb -> points[k].r = 255;
      remain_rgb -> points[k].g = 255;
      remain_rgb -> points[k].b = 255;

    }






## 点云配准

两个点云 source 和 target，找到一个变换将 source 变换尽量与 target 对齐

与（4）PCA 构建包围盒 那节的区别：构建包围盒那节是将**激光雷达坐标系1**下的点云转换到**点云重心坐标系2**下，并且是知道坐标系之间的关系，其关系是R12为主成分列向量矩阵的转置，t12为点云重心点在激光雷达坐标系1的坐标。

点云配准不知道两个坐标系之间的关系，想知道点云到点云的转换，求得点云转换矩阵后，坐标系的变换矩阵也知道了

**仿射变换：** 线性变换（旋转、缩放、剪切） + 平移，​保持直线和平行线​，允许形状和大小改变​（如缩放、拉伸、倾斜）

**刚体变换：** 仿射变换的子集，仅包含旋转 + 平移，保持物体形状和大小​（所有点之间的欧氏距离不变），不包含缩放或剪切

### 4PCS(4-Point Congruent Sets) 点云粗配准
**核心原理：**

（1）刚体变换不变性：四点构成的平面四边形中，边长比例和对角线比例在旋转和平移下保持不变

（2）​仿射不变性：四点集的交比（Cross Ratio）在仿射变换下不变，而刚体变换是仿射变换的特例。

**流程：**

(1) source 中选择四点基元，要求四点共面且两对边非平行

(2) 计算基元的仿射不变量，

(3) 在 target 中依据仿射不变量，且在误差范围内，找到与基础广域基近似全等的一致性四点集

(4) 通过基元与 target 中四点集计算刚性变换 T，根据重叠比例测试获得最佳变换 T

    pcl::console::TicToc time;
    pcl::registration::FPCSInitialAlignment<pcl::PointXYZ, pcl::PointXYZ> fpcs;
    fpcs.setInputSource(source);
    fpcs.setInputTarget(target);
    fpcs.setApproxOverlap(0.7);         // 设置近似重叠率
    fpcs.setDelta(0.01);                // 精度参数
    fpcs.setNumberOfSamples(100);       // 采样点数量
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
    fpcs.align(*aligned);               // 执行配准
    Eigen::Matrix4f final_transform = fpcs.getFinalTransformation(); // 获取变换矩阵
    // 输出变换矩阵
    std::cout << "变换矩阵：\n" << final_transform << std::endl;

**优点：**
无需初始位姿，直接从任意位置开始配准；对部分重叠（30%-70%）点云有效；多次采样不同基元，统计最优解，受噪声影响小

**缺点：**
要求四点基元共面，非平面结构配准困难；需遍历大量四点组合，时间复杂度高；交比容差和重叠率需精细调整；

**适用场景：**
​多视角碎片化数据​（如文物扫描、建筑BIM）；低重叠率点云粗对齐​（需后续ICP精配准）；






### K4PCS(4-Point Congruent Sets) 点云粗配准

(1) 先利用 voxel grid 下采样，再进行关键点检测  (3D Harris,3D DoG)

(2) 通过 4PCS 使用关键点进行匹配，降低搜索规模，提高运算效率


	// --------------------------K4PCS算法进行配准------------------------------
	pcl::registration::KFPCSInitialAlignment<pcl::PointXYZ, pcl::PointXYZ> kfpcs;
	kfpcs.setInputSource(source);  // 源点云
	kfpcs.setInputTarget(target);  // 目标点云
	kfpcs.setApproxOverlap(0.7);   // 源和目标之间的近似重叠。
	kfpcs.setLambda(0.5);          // 平移矩阵的加权系数。(暂时不知道是干什么用的)
	kfpcs.setDelta(0.002, false);  // 配准后源点云和目标点云之间的距离
	kfpcs.setNumberOfThreads(6);   // OpenMP多线程加速的线程数
	kfpcs.setNumberOfSamples(200); // 配准时要使用的随机采样点数量
	// kfpcs.setMaxComputationTime(1000);//最大计算时间(以秒为单位)。
	pcl::PointCloud<pcl::PointXYZ>::Ptr kpcs(new pcl::PointCloud<pcl::PointXYZ>);
	kfpcs.align(*kpcs);

	cout << "KFPCS配准用时： " << time.toc() << " ms" << endl;
	cout << "变换矩阵：\n" << kfpcs.getFinalTransformation() << endl;
	// 使用创建的变换对为输入的源点云进行变换
	pcl::transformPointCloud(*source, *kpcs, kfpcs.getFinalTransformation());
	// 保存转换后的源点云作为最终的变换输出
	pcl::io::savePCDFileBinary("transformed.pcd", *kpcs);
	





### RANSAC 点云粗配准
原理：通过随机采样和迭代验证，找到最优的刚体变换参数

**算法流程：**
（1）通过 FPFH 特征描述子提取关键点及其特征向量，对特征进行最近邻匹配，获得匹配点集M={(pi,qi)}

（2）使用匹配点集估计变换 T

（3）将估计变换应用于点云 P，与点云 Q 通过最近邻搜索寻找内点，内点太少则返回（1）

（4）内点对应关系重新估计 T，计算内点对应点之间的距离，达到最小距离则当前估计作为最终转换

pcl::SampleConsensusPrerejective 需要提前计算描述

    // RANSAC配准
    pointcloud::Ptr ransac_registration(pointcloud::Ptr source, pointcloud::Ptr target, fpfhFeature::Ptr source_fpfh, fpfhFeature::Ptr target_fpfh)
    {
        pcl::SampleConsensusPrerejective<PointT, PointT, pcl::FPFHSignature33> r_sac;
        r_sac.setInputSource(source);            // 设置源点云
        r_sac.setInputTarget(target);            // 设置目标点云
        r_sac.setSourceFeatures(source_fpfh);    // 设置源点云的FPFH特征
        r_sac.setTargetFeatures(target_fpfh);    // 设置目标点云的FPFH特征
        r_sac.setCorrespondenceRandomness(5);    // 随机特征对应时使用的邻居数量
        r_sac.setInlierFraction(0.5f);           // 设置所需的内点比例
        r_sac.setNumberOfSamples(3);             // 设置采样点的数量
        r_sac.setSimilarityThreshold(0.1f);      // 设置边缘长度相似度阈值
        r_sac.setMaxCorrespondenceDistance(1.0f);// 设置最大对应点距离
        r_sac.setMaximumIterations(100);         // 设置最大迭代次数

        pointcloud::Ptr aligned(new pointcloud); // 配准后的点云
        r_sac.align(*aligned);                   // 执行配准

        pcl::transformPointCloud(*source, *aligned, r_sac.getFinalTransformation()); // 对源点云进行变换
        cout << "变换矩阵：\n" << r_sac.getFinalTransformation() << endl;            // 输出变换矩阵

        return aligned;
    }

pcl::registration::RANSAC 内部计算描述特征

    #include <pcl/registration/ransac.h>
    #include <pcl/features/normal_3d_omp.h>

    pcl::registration::RANSAC<pcl::PointXYZ, pcl::PointXYZ> ransac;
    ransac.setInputSource(source_cloud);
    ransac.setInputTarget(target_cloud);
    ransac.setInlierThreshold(0.05);   // 内点残差阈值
    ransac.setMaxIterations(1000);     // 最大迭代次数
    ransac.align(*aligned_cloud);

**优点：**
抗离群点能力强（统计内点筛选最优变换T）；无需初始位姿：直接从随机采样开始，全局搜索最优解；

**缺点：**
高离群点比例时需极多迭代次数；需高质量特征描述子（如FPFH）生成候选点对；通常作为粗配准；

**应用场景：**
高噪声数据配准






### 点到点 ICP 点云精配准

**原理：**给定 source 点云 P 和 target 点云 Q，寻找刚体变换 T 使得误差距离最小

**算法流程：**

（1）设初始的位姿估计为 R0, t0

（2）从初始位姿估计开始迭代。设第 k 次迭代时位姿估计为 Rk, tk

（3）在 Rk, tk 估计下，按照最近邻方式寻找匹配点。记匹配之后的点对为 (pi, qi)

（4）计算迭代结果判断是否收敛，不收敛返回（3），收敛退出

$$
\underset{R, t}{\operatorname{argmin}} \sum_{i=1}^n\left\|\left(R p_i+t\right)-q_i\right\|^2
$$

可通过 SVD 求解最小二乘或高斯牛顿法优化求解


	//--------------------初始化ICP对象--------------------
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	//----------------------icp核心代码--------------------
	icp.setInputSource(source);            // 源点云
	icp.setInputTarget(target);            // 目标点云
	icp.setTransformationEpsilon(1e-10);   // 为终止条件设置最小转换差异
	icp.setMaxCorrespondenceDistance(1);  // 设置对应点对之间的最大距离（此值对配准结果影响较大）。
	icp.setEuclideanFitnessEpsilon(0.001);  // 设置收敛条件是均方误差和小于阈值， 停止迭代；
	icp.setMaximumIterations(35);           // 最大迭代次数
	icp.setUseReciprocalCorrespondences(true);//设置为true,则使用相互对应关系
	// 计算需要的刚体变换以便将输入的源点云匹配到目标点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	icp.align(*icp_cloud);
	cout << "Applied " << 35 << " ICP iterations in " << time.toc() << " ms" << endl;
	cout << "\nICP has converged, score is " << icp.getFitnessScore() << endl;
	cout << "变换矩阵：\n" << icp.getFinalTransformation() << endl;
	// 使用创建的变换对为输入源点云进行变换
	pcl::transformPointCloud(*source, *icp_cloud, icp.getFinalTransformation());
	//pcl::io::savePCDFileASCII ("666.pcd", *icp_cloud);

**优点：**
精度高；KD-Tree加速后适用于中等规模点云；

**缺点：**
需粗配准（如4PCS或RANSAC）提供初始变换；易陷入局部最优；对低重叠率（<30%）点云失效；噪声敏感需预处理

**适用场景：**
相邻帧点云配准；​扫描数据对齐







### kdtree 优化 ICP

	//--------------------初始化ICP对象--------------------
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

	//---------------------KD树加速搜索--------------------
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZ>);
	tree1->setInputCloud(source);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
	tree2->setInputCloud(target);
	icp.setSearchMethodSource(tree1);
	icp.setSearchMethodTarget(tree2);
	//----------------------icp核心代码--------------------
	icp.setInputSource(source);            // 源点云
	icp.setInputTarget(target);            // 目标点云
	icp.setTransformationEpsilon(1e-10);   // 为终止条件设置最小转换差异
	icp.setMaxCorrespondenceDistance(1);  // 设置对应点对之间的最大距离（此值对配准结果影响较大）。
	icp.setEuclideanFitnessEpsilon(0.05);  // 设置收敛条件是均方误差和小于阈值， 停止迭代；
	icp.setMaximumIterations(35);           // 最大迭代次数
	//icp.setUseReciprocalCorrespondences(true);//使用相互对应关系
	// 计算需要的刚体变换以便将输入的源点云匹配到目标点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	icp.align(*icp_cloud);
	cout << "Applied " << 35 << " ICP iterations in " << time.toc() << " ms" << endl;
	cout << "\nICP has converged, score is " << icp.getFitnessScore() << endl;
	cout << "变换矩阵：\n" << icp.getFinalTransformation() << endl;
	// 使用创建的变换对为输入源点云进行变换
	pcl::transformPointCloud(*source, *icp_cloud, icp.getFinalTransformation());
	//pcl::io::savePCDFileASCII ("666.pcd", *icp_cloud);





## 点云分割
### 欧式聚类分割
比较适用于没有连通性的点云分割聚类

优点：
1. 无需计算法线或颜色属性，适合实时处理
2. 准确分割空间中明显分离的物体
3. 不受颜色或材质差异影响，仅依赖几何分布
  
缺点：
1. 如紧挨着的物体会被合并
2. 距离阈值需根据场景调整


**流程：**
1. 选取空间中一点 p
2. kd tree 最近邻搜索
3. 距离小于阈值的点放入 Q 类中
4. 判断 Q 中点数量是否增加，不再增加则结束，继续增加则在 Q 中选取 p 之外的点
5. 重复 1 到 4


### 区域生长分割
优点：
1. 适合分割具有相同表面的区域，如连续平面

缺点：
1. 需计算法线、曲率等属性
2. 法线夹角阈值、曲率阈值等需精细调优
   
**流程：**
1. 选取曲率最小的点（如平面区域）作为初始种子
2. 计算近邻点与种子点的法线夹角。夹角 < 阈值，进入下一步；夹角 > 阈值，重新找近邻点
3. 计算点云曲率。曲率 < 阈值，加入种子点的点集合；曲率 > 阈值，重新找近邻点


<!-- ## 1.3 PCA
### normals
code: [normals.cpp](src/PCL_learn/pca/normals.cpp)

求点云法向量，遍历点，对于某个点，找这个点的最近邻十个点，将这十个点放入 pca 中得到一个 eigen_vectors，最后一列（最小特征值对应的）为法向量

或使用 pcl::NormalEstimation\<pcl::PointXYZ, pcl::Normal\> 

    method1:
    pcl::PCA<pcl::PointXYZRGB> pca;
    pca.setInputCloud(cloud_);
    Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();

    method2:
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal>




### 点云 PCA
code: [pca.cpp](src/PCL_learn/pca/pca.cpp)

求点云主成分，特征值最小的对应的向量是法向量，这里是纯 eigen 算，也可以调用 pcl 求 -->


## 1.5 特征点与特征描述

### 1.5.1 pcl::Feature
1. setIndices() = false, setSearchSurfase() = false
   
   仅在 input cloud 中搜索最近邻，最常见的情况
2. setIndices() = true, setSearchSurfase() = false

   有索引，仅计算索引点云的特征，knn时可以使用索引以外的点
3. setIndices() = false, setSearchSurfase() = true

   无索引，有指定搜索面，计算所有输入点云的特征，但是寻找最近邻只能在 SearchSurfase 点云中寻找

   在点云非常密集的情况下，输入点云为下采样点云，SearchSurfase 为原始点云，这样计算特征效率高
4. setIndices() = true, setSearchSurfase() = true

   这种情况比较少见，计算 input 的 Indices 点特征，knn 从 SearchSurfase 找


### 1.5.2 点云 PCA

**PCL 求法向量**

code: [normals.cpp](src/PCL_learn/pca/normals.cpp)

求点云法向量，遍历点，对于某个点，找这个点的最近邻十个点，将这十个点放入 pca 中得到一个 eigen_vectors，最后一列（最小特征值对应的）为法向量

或使用 pcl::NormalEstimation\<pcl::PointXYZ, pcl::Normal\> 

    method1:
    pcl::PCA<pcl::PointXYZRGB> pca;
    pca.setInputCloud(cloud_);
    Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();    
    method2:
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal>

**eigen 求点云法向量**

code: [pca.cpp](src/PCL_learn/pca/pca.cpp)

求点云主成分，特征值最小的对应的向量是法向量，这里是纯 eigen 算，也可以调用 pcl 求，但是需要法向量定向过程


### 1.5.3 求点云曲率
**主曲率：**
1. ​第一主曲率（κ1​）​：最大曲率值，沿曲率最大的方向。
2. ​第一主曲率（κ2）​：最小曲率值，沿曲率最小的方向。
   
**高斯曲率：**

K = κ1κ2

**平均曲率：**

H = (κ1 + κ2)/2

**表面曲率：**

表面曲率是点云数据表面的特征值来描述点云表面变化程度的一个概念，与数学意义上的曲率不同。

在计算法向量的时候，计算均值与协方差矩阵，再计算协方差矩阵的三个特征值（λ1 λ2 λ3）和特征向量

δ = λ1/(λ1 + λ2 + λ3) λ1最小

δ 越小周围越平坦，越大周围起伏越大


**PCL计算曲率的两个方法：**
1. pcl::PrincipalCurvaturesEstimation 输出 κ1 κ2
2. pcl::NormalEstimation 输出 κ2



### 1.5.4 MLS 平滑点云并计算法向量
MLS（移动最小二乘法） 可以用于平滑点云，并计算法向量，可以只用来计算法向量

在 MLS 第一步计算的超平面法向量作为采样点的法向量，此方法精度相对于 PCA 求法向量精度更高但是更消耗资源


    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
    mls.setInputCloud(cloud);
    mls.setComputeNormals(true);     // 是否计算法线，设置为ture则计算法线
    mls.setPolynomialOrder(2);       // 设置MLS拟合的阶数，默认是2
    mls.setSearchMethod(tree);       // 邻域点搜索的方式
    mls.setSearchRadius(0.005);      // 邻域搜索半径
    mls.setNumberOfThreads(4);       // 设置多线程加速的线程数
    mls.process(*mls_points_normal); // 曲面重建


### 1.5.5 计算点云包围盒
**(1) AABB 包围盒**

原理：
直接取点云 xyz 中最小值和最大值

特点：
包围盒XYZ轴都与坐标轴对齐

适用场景：
快速碰撞检测、空间索引

      pcl::PointXYZ minPt, maxPt;
      pcl::getMinMax3D(*cloud, minPt, maxPt);
      viewer->addCube(minPt.x, maxPt.x, minPt.y, maxPt.y, minPt.z, maxPt.z, 1.0, 0.0, 0.0, "AABB");

**(2) 惯性矩获得 AABB**

特点：
包围盒XYZ轴都与坐标轴对齐

      pcl::MomentOfInertiaEstimation<pcl::PointXYZ>mie;
      mie.setInputCloud(cloud);
      mie.compute();
      pcl::PointXYZ minPt, maxPt;
      mie.getAABB(minPt, maxPt); 

**（3）OBB 有向包围盒**

原理：PCA 获得主方向，在主方向坐标系中计算xyz最大值与最小值，得到包围盒

特点：
包围盒XYZ轴都紧贴物体

适用场景：
精确碰撞检测、物体姿态估计

      pcl::MomentOfInertiaEstimation<pcl::PointXYZ> mie;
      mie.setInputCloud(cloud);
      mie.compute();
      float maxValue, midValue, minValue;                // 三个特征值
      Eigen::Vector3f maxVec, midVec, minVec;            // 特征值对应的特征向量
      Eigen::Vector3f centroid;                          // 点云质心
      pcl::PointXYZ minPtObb, maxPtObb, posObb;          // OBB包围盒最小值、最大值以及位姿
      Eigen::Matrix3f rMatObb;                           // OBB包围盒对应的旋转矩阵
      mie.getOBB(minPtObb, maxPtObb, posObb, rMatObb);   // 获取OBB对应的相关参数
      mie.getEigenValues(maxValue, midValue, minValue);  // 获取特征值
      mie.getEigenVectors(maxVec, midVec, minVec);       // 获取特征向量
      mie.getMassCenter(centroid);                       // 获取点云中心坐标


**（4）PCA 构建包围盒**

获取质心（均值）和协方差矩阵，对协方差矩阵进行分解获得坐标系2，获得R12（坐标系1到坐标系2的旋转）

质心的坐标为 t12

已知1坐标系的点 P1 求2坐标系下点坐标

p1 = R12*p2 + t12

(R12)^-1(p1-t12) = p2




### 1.5.6 点云边界提取

选取一点P，knn找最近邻点，求该点法向量，找到切平面，将其knn点集投影到切平面，计算P到其他点的向量
夹角，当夹角阈值大于 pi/2 则 P 为边界点

<img src="note_pic/19.png"  width="500" />



	
    // ----------------------------边界特征估计------------------------------
    pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundEst;
    boundEst.setInputCloud(cloud);
    boundEst.setInputNormals(normals);
    boundEst.setRadiusSearch(0.02);
    boundEst.setAngleThreshold(M_PI / 2); // 边界判断时的角度阈值
    boundEst.setSearchMethod(tree);
    pcl::PointCloud<pcl::Boundary> boundaries;
    boundEst.compute(boundaries);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_boundary(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < cloud->points.size(); i++)
    {
      if (boundaries[i].boundary_point > 0)
      {
        cloud_boundary->push_back(cloud->points[i]);
      }
    }
    cout << "边界点个数:" << cloud_boundary->points.size() << endl;


### 1.5.7  Alpha Shapes 平面点云边界特征提取

**原理：**
任意形状的平面点云，一个半径为 α 的圆进行滚动，α 合适时圆只在边界滚动，滚动轨迹为点云边界

**算法流程：**
1. 对于任意一点 P，滚动半径 α，在点云内寻找 2α 以内的点组成集合 Q 
2. Q 中任意取一点 P1，计算 α 为半径，经过 P 和 P1 的两个圆的圆心 P2 P3
3. Q中去除 P1，在 Q 中寻找其他点，若其他点到 P2 和 P3 的距离均大于 α 则 P 为一个边界点
4. 若其他点到 P2 和 P3 距离不都大于 α，则 Q 中点轮换作为 P1。若某一点满足(2)(3)的条件，则该点为边界点，终止判断，若不存在这样的 P1 则 P 点为平面点

<img src="note_pic/20.jpg"  width="300" />



    // -------------------------a-shape平面点云边界提取-----------------------------
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudHul(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ConcaveHull<pcl::PointXYZ> chull; // a-shape
    chull.setInputCloud(cloud);            // 输入点云为投影后的点云
    chull.setAlpha(0.1);                   // 设置alpha值为0.1
    chull.reconstruct(*cloudHul);
    cout << "提取边界点个数为: " << cloudHul->points.size() << endl;
    cout << "提取边界点用时： " << time.toc() / 1000 << " 秒" << endl;
    pcl::PCDWriter writer;
    writer.write("hull.pcd", *cloudHul, false);









# 2. Lidar 运动补偿
https://blog.csdn.net/brightming/article/details/118250783

https://blog.csdn.net/qq_30460905/article/details/124919036

**代码：**~/PointCloud/PointCloud_ws/src/motion_compensation

## 点云预处理
加载 livox 头文件，使用 livox CustomMsg 格式点云

    cpp：
    #include <livox_ros_driver/CustomMsg.h>

    CMakeLists.txt:
    find_package(catkin REQUIRED COMPONENTS
    livox_ros_driver
    )

    package.xml:
    <build_depend>livox_ros_driver</build_depend>
    <exec_depend>livox_ros_driver</exec_depend>
## 使用 IMU 进行补偿
参考 Fast-LIO 中的函数

    void ImuProcess::UndistortPcl
    (const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
### 时间戳问题
livox 格式点云每个点都有自己的时间戳，官方驱动时间从启动为0开始，且雷达内置 imu 也如此；r3live的驱动时间戳为ros time，需要可以使用r3live的驱动

    pl_full[i].curvature = (msg->timebase + msg->points[i].offset_time) / float(1000000);
这里不能简单替换为 ros::Time::now() ！！！

使用官方驱动时，启动 livox 驱动的 ros time 为 T1，启动后第一帧点云时间戳为 0，dt = T1；在其他算法中这个 T1 我们不知道，无法将点云时间戳同步到 rostime。








# 3. 点云配准

# 4. 深度学习基础

动手学深度学习：https://zh-v2.d2l.ai/

## 4.1 多层感知机 (MLP)
可以通过在网络中加入一个或多个隐藏层来克服线性模型的限制， 这种架构通常称为多层感知机（multilayer perceptron），通常缩写为 MLP

![alt text](note_pic/2.png)

这个多层感知机有4个输入，3个输出，其隐藏层包含5个隐藏单元，输入层不涉及计算，隐藏层和输出层有计算，这个 MLP 的层数为2

多层感知机在输出层和输入层之间增加一个或多个全连接隐藏层，并通过激活函数转换隐藏层的输出。

### 激活函数
将神经网络非线性化，如果不用激活函数，每一层输出都是上层输入的线性函数，无论神经网络有多少层，输出都是输入的线性组合

常用激活函数：ReLU、sigmoid、tanh

### 训练误差与泛化误差
**训练误差（training error）**：模型在训练数据集上计算得到的误差

**泛化误差（generalization error）**：模型应用在同样从原始样本的分布中抽取的无限多数据样本时，模型误差的 ***期望*** 。说白了就是在训练集上没见过的数据的错分样本比率？
### 训练集、验证集、测试集
常见做法是将我们的数据分成三份：
1. **训练集**：训练集用来训练模型，即确定模型的权重和偏置这些参数
2. **验证集**：验证集用于模型的选择，比较具有不同数量的隐藏层、不同数量的隐藏单元以及不同的激活函数组合的模型
3. **测试集**：测试集只使用一次，即在训练完成后评价最终的模型时使用

**K 折交叉验证：**

当训练数据稀缺时, 原始训练数据被分成 K 个不重叠的子集。 然后执行 K 次模型训练和验证

每次在 K-1 个子集上进行训练， 并在剩余的一个子集（在该轮中没有用于训练的子集）上进行验证

### 欠拟合与过拟合
**欠拟合**：训练误差和验证误差都很严重， 但它们之间仅有一点差距

**过拟合**：训练误差明显低于验证误差时要小心

<!-- ![alt text](3.png) -->
<img src="note_pic/3.png"  width="350" />

### 权重衰减
**目的**：限制模型复杂度，抑制模型的过拟合，提高模型的泛化性

**方法**：在训练集的损失函数中加入惩罚项，以降低学习到的模型的复杂度， 将原来的训练目标最小化训练标签上的预测损失， 调整为最小化预测损失和惩罚项之和

正常的损失函数：

<img src="note_pic/image-2.png"  width="300" />

加入一个额外的损失 **（权重的L2范数）** 来限制权重向量的大小，通过 **正则化常数 λ** 平衡这个新的额外惩罚的损失

<img src="note_pic/image-3.png"  width="170" />

https://blog.csdn.net/zhaohongfei_358/article/details/129625803

### 前向传播、反向传播、计算图

**前向传播**：按顺序（从输入层到输出层）计算和存储神经网络中每层的结果。

输入样本为 $\mathbf{x}$ 中间变量为 $\mathbf{z}$，$\mathbf{W}^{(1)}$ 为隐藏层的权重参数
$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x}$$
激活函数为 $\phi$
$$\mathbf{h}= \phi (\mathbf{z})$$
输出层的参数只有 $\mathbf{W}^{(2)}$
$$ \mathbf{o}= \mathbf{W}^{(2)} \mathbf{h} $$
损失函数 l 样本标签 y
$$L = l(\mathbf{o}, y)$$
正则化项（权重衰减）为
$$ s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right)$$
最后，模型在给定数据样本上的正则化损失为
$$J = L + s.$$
将 $J$ 称为目标函数（objective function）

前向传播的计算图如下，其中正方形表示变量，圆圈表示操作符。左下角表示输入，右上角表示输出

<img src="./note_pic/forward.png"  width="480" />

**反向传播**：（backward propagation或backpropagation）指的是计算神经网络参数**梯度**的方法

在上面计算图中，反向传播用于计算梯度 $\partial J/\partial \mathbf{W}^{(1)}$ 和 $\partial J/\partial \mathbf{W}^{(2)}$ 应用链式法则，依次计算每个中间变量和参数的梯度

### 参数初始化
Xavier 初始化可以避免梯度消失和梯度爆炸

## 4.2 深度学习计算
### 层和块
块（block）可以描述单个层、由多个层组成的组件或整个模型本身
 
块负责大量的内部处理，包括参数初始化和反向传播

### 参数管理、读写文件

## 4.3 卷积神经网络
<img src="note_pic/4.png"  width="350" />

卷积层对输入和卷积核权重进行互相关运算，并在添加标量偏置之后产生输出

卷积层中的两个被训练的参数是卷积核权重和标量偏置

感受野：指特征图上的某个点能看到的输入图像的区域，特征图上的点是由输入图像中感受野大小区域的计算得到的

### 填充和步幅
在应用多层卷积时，我们常常丢失边缘像素，在周围填充一圈0

卷积神经网络中卷积核的高度和宽度通常为奇数，保持空间维度的同时，可以在顶部和底部填充相同数量的行，在左侧和右侧填充相同数量的列

步幅分为水平步幅和垂直步幅

填充和步幅可用于有效地调整数据的维度

### 多通道输入输出

多通道输入：需要构造一个与输入数据具有相同输入通道数的卷积核：两个通道则需要两个卷积核

<img src="note_pic/5.png"  width="400" />

多通道输出：

1*1 卷积层通常用于调整网络层的通道数量和控制模型复杂性

## 4.4 Transformer

**encoder - decoder:** 

将输入编码为向量，n 个输入（x1 ... xn）编码为 n 个向量（z1 ... zn），解码器将 n 个向量生成 m 个输出（z1 ... zm）。对于编码器来说，可以看到整个输入一次性编码；解码器只能一个一个生成，输出 y <sub>2</sub> 需要 y <sub>1</sub>，之前的输出也作为当前时刻的输入。

**attention：**
mask 作用是防止 t 时刻看到以后的东西

## 4.5 注意力机制
权重的分布

<img src="note_pic/7.png"  width="450" />

**标量注意力：**
其中每个注意力权重是一个标量值

**向量注意力：**
其中每个注意力权重是一个向量，在更细粒度的水平上对不同的维度进行加权

## 4.6 nn.Conv1d
https://cloud.tencent.com/developer/article/2061820

    torch.nn.Conv1d(in_channel, dim, 1)

对一个序列做卷积，当卷积核为 2 时，序列变短了，当卷积核为 1 时，序列长度不变

dim 表示输出的特征维度，相当于一个全连接层，可以进行维度的升高与降低

<img src="note_pic/17.png"  width="350" />

## 4.7 nn.Conv2d

    nn.Conv2d(in_dim, out_dim, kernel_size)

    input: torch.randn(5,in_dim,20,3)
    nn.Conv2d(in_dim, out_dim, 1)
    out put: torch.Size([5, out_dim, 20, 3])

卷积核大小为 1，对于图像来说，不改变 H W，也就是后两个维度，只改变特征通道数

## 4.8 nn.BatchNorm1d

    points66 = torch.tensor([[[1, 2, 3, 3],
                              [4, 2, 2, 7],
                              [1, 6, 3, 4]],
                      
                             [[6, 1, 3, 1],
                              [4, 8, 0, 5],
                              [1, 0, 5, 4]]]) # [2, 3, 4] [batch, Feature, N] 这样的格式cv用的多
    points66 = points66.float()
    bn = nn.BatchNorm1d(3)     # 输入张量的特征维度
    points_bn = bn(points66)

Batch Norm 对于不同的 Batch 的同一个特征做归一化，batch1 的第一行为4个点的x值，batch2 的第一行为4个点的x值

bn是对于特征操作，计算过程：

第一个 batch 中 [1, 2, 3, 3] 和 第二个 batch 中 [6, 1, 3, 1] 求均值 μ 和方差 σ^2, 

输出第一个 batch 第一行：(1-μ) / (σ^2-ε)^0.5 , (2-μ)/(σ^2-ε)^0.5, (3-μ)/(σ^2-ε)^0.5, (3-μ)/(σ^2-ε)^0.5

输出第二个 batch 第一行：(6-μ) / (σ^2-ε)^0.5, (1-μ)/(σ^2-ε)^0.5, (3-μ)/(σ^2-ε)^0.5, (1-μ)/(σ^2-ε)^0.5

## 4.9 nn.LayerNorm

    points66 = torch.tensor([[[1, 2, 3, 3],
                              [4, 2, 2, 7],
                              [1, 6, 3, 4]],

                             [[6, 1, 3, 1],
                              [4, 8, 0, 5],
                              [1, 0, 5, 4]]]) # [2, 3, 4] [batch, N, Feature] 这种格式NLP用的多
    points66 = points66.float()
    bn = nn.LayerNorm(normalized_shape=[4])

LayerNorm 是对一个样本（batch）进行操作，与另一个batch无关 **normalized_shape=[4]** 表示只对最后一个维度（特征）操作

第一个 batch 中 [1, 2, 3, 3] 均值：2.25   和方差 σ^2 = 0.6875

输出第一个 batch 第一行：(1-μ) / (σ^2-ε)^0.5 , (2-μ) / (σ^2-ε)^0.5, (3-μ)/(σ^2-ε)^0.5, (3-μ)/(σ^2-ε)^0.5



    points66 = torch.tensor([[[1, 2, 3, 3],
                              [4, 2, 2, 7],
                              [1, 6, 3, 4]],
                      
                             [[6, 1, 3, 1],
                              [4, 8, 0, 5],
                              [1, 0, 5, 4]]]) # [2, 3, 4] 
    points66 = points66.float()
    bn = nn.LayerNorm(normalized_shape=[3, 4])

**normalized_shape=[3, 4]** 对 3*4 12个值进行归一化

求 [1, 2, 3, 3, 4, 2, 2, 7, 1, 6, 3, 4] 的均值和方差 

    mean1 = points66.mean()

    var1 = points66.var(unbiased=False)




#  5. 点云语义分割
<!-- <img src="note_pic/image.png"  width="600" />
<img src="note_pic/image-1.png"  width="600" /> -->

分类 目标检测 语义分割 区别
语义分割给每个像素一个 label

## 5.1 Point Net
在 Point Net 之前，将点云转换为栅格，用 3D CNN 处理，但是分辨率降低；或使用投影的方法，用 2D CNN 处理。

对于点云数据，**网络输入是无序的** 且 **不同视角结果应该是一样的**

对于无序数据，对称函数不用在意输入数据的顺序，比如 max 函数，point net 的核心思想是构造一个复合函数，其中一层是对称函数，整个网络也就是对称函数

<img src="note_pic/11.png"  width="800" />

分类网络：
1. 输入点云通过一个 input transform 转换视角，T-Net 为 3 * 3 的矩阵，与输入进行矩阵乘法，得到的还是n*3 的矩阵
2. 每个点通过同一个 mlp 扩展维度，3维到64维
3. 再通过一个特征转换视角的模块（Point Net++ 去掉了）
4. 通过 mlp 升维到 1024 维，n * 1024
5. 经过一个 max pool 得到 1*1024 的向量
6. 通过一个 mlp 变成 k 维，1*k 表示这组点云在每一个类别上的得分

<img src="note_pic/12.png"  width="800" />

分割网络（每个点都有 k 个得分）：

1. 将每个点的局部特征和全局特征拼接在一起，变成了 n*(64 + 1024)
2. 进行 mlp 变成 1088 维，再通过 mlp 变成 n*m 的得分

**从 global feature 可以获得点云的 critical points**

## 5.2 Point Net++
Point Net 中，一个点自己经过mlp扩展特征维度，Point Net++通过点与其周围的点进行扩展维度

<img src="note_pic/13.png"  width="500" />

**sampling -> grouping -> pointnet**

sampling: uniform sampling, FPS

grouping: KNN, Ball query

<img src="note_pic/14.png"  width="1000" />

1. 对点云进行采样与聚合，经过一个 point net 点数量减少，特征维度增加
2. 再进行上面的过程，获得红色的数据
3. 分类：红色的全局特征经过point net 和 mlp 得到分类得分
4. 分割：红色的全局特征经过插值，再与之前蓝色的数据进行拼接，经过point net再插值拼接，获得每个点的得分

在 Point Net++ 中，grouping 环节受到点密度的影响，离激光雷达近的地方点密度大，远的地方密度小，在远的地方用球采样，可能导致球里面点很少，影响特征提取。文章中提出 **MSG** 和 **MRG** 解决，MSG 在同一级别上用不同大小的半径提取特征，并进行拼接；MRG 在不同级别上提取特征进行拼接。

<img src="note_pic/15.png"  width="500" />

在分割任务中，需要恢复点的数量，找要恢复的点最近的三个上层点，使用距离的倒数作为权重进行插值，再将原来的特征拼接再后面

## 5.3 RangeNet++
其核心思想是将 ​​3D 点云投影为 2D 距离图像（Range Image），利用 2D 卷积网络进行特征提取，再通过后处理优化分割结果

算法流程：

1. 点云转换深度图，(x,y,z)转换为极坐标再进行投影，转换为 (u,v)
2. 2D语义分割
3. 从2D到3D的语义转移，从原始点云中恢复所有点
4. 解决投影离散化导致同一像素内多个点的标签冲突


# 6. 点云补全
## partial to complete
[VQ VAE 介绍](https://zhuanlan.zhihu.com/p/633744455)

[VAE 介绍](https://zhuanlan.zhihu.com/p/574208925)

问题：模型推理的过程中，VQ VAE解码器的输入是什么

## 配 EditVAE 环境
conda 环境名：editvae2

1. 安装 [PyTorchEMD](https://github.com/daerduoCarey/PyTorchEMD)
  
   环境配置：torch version: 1.13.0+cu117      CUDA version: 11.7
   
   在 cuda/emd_kernel.cu 中：
   
   1. 注释掉 #include <THC/THC.h> 
   
   2. Replace all THCudaCheck with C10_CUDA_CHECK
   
   3. CHECK_EQ(a, b); -> TORCH_CHECK_EQ(a, b);
   
   4. AT_CHECK(a == b, "Error message"); -> TORCH_CHECK(a == b, "Error message");

    编译的时候使用 python setup.py install 报错 没有权限，sudo python setup.py install 也不行，没有命令，使用 sudo python3 setup.py install 报错没有 torch，原因是 sudo 用的 base 的环境，不是自己的 conda 环境。最后使用 sudo -E python3 setup.py install 成功编译，sudo -E 保留当前用户的环境变量

    配置完 PyTorchEMD 并通过测试

2. 装完之后要将 emd_cuda.cpython-37m-x86_64-linux-gnu.so 文件放在 utils 下，一定要用 python3.7 编译PyTorchEMD， 后面发现还有一个库 fast_sampler 是用 python3.7 完成编译的。
但是使用 sudo -E python3 setup.py install 编译python环境又变成了 3.8，应该使用 ：   
      sudo -E /home/ps/anaconda3/envs/editvae_3.7/bin/python3.7 setup.py install
3. 报错：

   <img src="note_pic/16.png"  width="700" />

   measurement.py 473行 gts = np.concatenate((gts, gt), axis=1) 有 bug，直接注释了





## AnchorFormer
传统方法：输入点云 --> 全局特征向量 --> 稠密点云 **池化操作会导致点云细节缺失**

AnchorFormer：输入点云 --> Anchors --> 稀疏点云 --> 稠密点云

1. 通过基于输入部分观察的点特征学习一组 **锚点** 来模拟区域区分
2. **锚点** 通过估计特定的偏移量来分散到观察到的位置和未观察到的位置并与输入观测的下采样点形成稀疏点云
3. 为了获得稠密点云，将稀疏点各个位置的规范 2D 网格变形为详细的 3D 结构

<img src="note_pic/6.png"  width="500" />

锚点可以推断观察到的点的关键模式还能表示缺失的部分，将输入观测的锚点和下采样的点作为稀疏点，再扩充成稠密点

1. 首先对输入点进行下采样，并通过基于 EdgeConv 的头部提取点特征
2. Transformer 编码器将下采样点的点特征作为输入，并学习在编码器的每个基本块中预测一组坐标，即锚点
3. 同时，下采样点和锚点的点特征也通过编码器进行细化、
4. 通过学习特定的偏移量，锚点进一步分散到不同的3D位置
5. 最后，AnchorFormer 将下采样点和锚点组合为稀疏点

### Anchor Generation
**特征提取：**

1. 下采样：FPS 最远点 得到 S <sub>0</sub>
2. 特征提取：EdgeConv 得到 F <sub>0</sub>

**锚点预测：**

双重注意力模块为 transformer 编码器，
当前的双重注意块预测一组新的锚点，同时细化前一个块的输入点特征

为了表征未观察部分，提出**特征扩展模块**：
1. 利用输入点特征与相应池化特征向量之间的特征差异进行锚点特征预测。
2. 利用预测的锚点特征和输入点特征之间的交叉注意进行锚点坐标学习。

流程：
1. 通过自注意力机制增强特征；F <sub>i-1</sub> --> X <sub>i</sub>
2. 池化 X <sub>i</sub> 得到 g <sub>i</sub>；X <sub>i</sub> --> g <sub>i</sub>
3. 对增强点特征 X<sub>i</sub> 与相应池化特征向量 g <sub>i</sub> 之间的特征差进行线性投影，得到锚点特征；X <sub>i</sub><sup>'</sup> = MLP ( g<sub>i</sub> - X<sub>i</sub> )
4. 增强点特征  X<sub>i</sub>、预测的锚点特征 X <sub>i</sub><sup>'</sup>、输入点 S <sub>i-1</sub> 通过交叉注意力机制得到预测的锚点坐标 a <sub>i</sub>


### Anchor Scattering
锚点 + 输入的下采样点 --> 丰富稀疏点细节

缺失部分的空间没有足够的锚点来促进详细的结构重建，通过学习特定的偏移量 ∆A  来将锚点分散到不同的位置

分散后的锚点 A′ = A + ∆A

稀疏点云 S：分散后的锚点 和 输入残缺点云下采样的点

### Point Morphing

## Point Transformer
**自注意力网络在 3D 点云处理中的应用**

基于投影的网络：将点云投影到各个平面，没有充分利用点云的稀疏性，影响 3D 中的识别性能和遮挡可能会阻碍准确性

基于体素的网络：计算量大

基于点的网络：设计了直接摄取点云的深度网络结构

基于 Transformer 和 自注意力：3D 点云本质上是具有位置属性的点集，自注意力机制似乎特别适合这种类型的数据，局部应用  selfattention

### Point Transformer Layer

使用的是向量注意力权重

### Position Encoding

δ = θ ( p <sub>i</sub> − p <sub>j</sub> ) 

p <sub>i</sub> 和 p <sub>j</sub> 为三维坐标，编码函数 θ 是一个具有两个线性层和一个 ReLU 非线性的 MLP

### Point Transformer Block

### Network Architecture


## SeedFormer
贡献：
1. 引入了一种新的形状表示，即Patch Seed，它不仅从部分输入中捕获一般结构，而且还保留了局部模式的区域信息
2. 设计了一种新的点生成器，即上采样 Transformer，通过将 Transformer 结构扩展到生成点的基本操作中。

解码阶段由两个主要步骤组成:
1. 首先从种子生成器中的不完整特征生成完整的形状
2. 然后以粗到细的方式恢复细粒度的细节

### Architecture Overview
<img src="note_pic/8.png"  width="750" />

**Encoder：**

输入点云，使用 point transformer 和 abstraction layers 从残缺点云中提取特征，每往下一层点的数量逐渐减少，得到 patch 特征（F<sub>p</sub>）和 patch 中心坐标（P<sub>p</sub>），表示点云的部分结构

**Seed generator：**

生成一个粗略但完整的点云（seed points）以及每个点的种子特征；

给定提取的 patch 特征 F<sub>p</sub> 和中心坐标 P<sub>p</sub>，使用Upsample Transformer 生成一组新的种子特征 F，F 通过 MLP 生成相应的 seed points。

**Coarse-to-fine generation：**

使用Upsample Transformer将输入点云中的每个点上采样到 r<sub>l</sub> 个点，馈送到第一层的粗略点云 P0 是通过使用最远点采样(FPS)融合种子S和输入点云P来生成的

### Point Cloud Completion with Patch Seeds
**Patch Seeds:**

由种子坐标 S 和特征 F 组成，每个种子覆盖该点周围的一个小区域


### 确定一个 seedformer 测试集

ShapeNet-55 的数据格式：(8192, 3)，将 ShapeNet-55 test.txt 中的数据弄成 partial points (2048, 3)

## PointAttN

**几何细节感知单元(GDP):**

对 x<sub>i</sub> 进行 knn 得到最近邻点，根据 k 个最近邻点，建立点 x<sub>i</sub> 的特征，**接收域有限**。

使用交叉注意来建立输入点云特征与其下采样点云特征之间的点关系。

KNN 只是提取 x<sub>i</sub> 的最近邻点，而真正与 x<sub>i</sub> 相关的点可能不在最近的点里面

<img src="note_pic/9.png"  width="650" />

**自特征增强单元(SFA):** 

自注点意力建立输入点云中点之间的关系，允许云中的每个点特征来增强其全局感知能力，来预测完整云

SFA 接收输入 (X, u)，其中 X 是 n × c 的矩阵，u 是上采样比，SFA 的输出是一个大小为 n × uc 的矩阵


## WalkFormer

基于特征相似度对局部主导点进行采样，并移动点形成缺失的部分

### Point Walk
**Neighbour Similarity Sampling：**

选取要移动的点，FPS 等下采样方法会导致从具有独特几何信息的外部区域中选择点

邻域相似性采样：先用 FPS 获得相对均匀的质心，得到 i 个质心，通过球查询算法得到每个质心半径内的 K 个点，算一个质心球中找一个点使特征余弦相似度最大，这个点作为一个 walk 起始点

**Point Selector：**

点进行 walk，walk 过程定义为一个序列 ω ，用 π(·) 来指导行走过程中下一个点的选择。


## Zero-shot
对于给定的部分点云Pin，我们首先将其转换为带点云着色的参考图像Iin和3D高斯Gin，将Iin和Gin引入到 Zero-shot 分形补全中,生成完成的点云Pout

<img src="note_pic/10.png"  width="850" />

### Point Cloud Colorization:

通过 reference viewpoint estimation 获得相机位姿 Vp，Pin 初始化 3DGS 得到 Gin，Gin的中心是固定的，以保持Pin的形状。

## CDPNet (AAAI 2024)
点云补全跨模态双相位网络

形状的全局信息是从额外的单视图图像中获得的，部分点云提供几何信息

### Introduction:
**Challenge:**

如何利用多模态特征来完成形状补全  如何生成形状的细节。

**Pipeline:**

Phase1: Image -> coarse point clouds   segment the coarse point clouds into patches and generate dense point clouds

Phase2: 通过DGCNN从partial点云中提取细粒度的几何信息   粗局部几何信息与细粒度几何信息结合，发送到多补丁生成器，获得的稠密点云和partial相加再FPS(MSN (Liu et al. 2020))

**contributions:**

1. 利用图像学习全局信息，利用patches保留局部几何细节
2. design a new patch generator  接收粗糙的patch特征和细粒度的几何信息来生成细粒度patch
3. CDPNet




# 7. Conda
## 解决 conda 权限问题
报错：

    EnvironmentNotWritableError: The current user does not have write permissions to the target environment.
      environment location: /home/ps/anaconda3/envs/Seedformer
      uid: 1000
      gid: 1000

解决：

    sudo chmod -R 777 /home/ps/anaconda3/envs/Seedformer

可以正常安装：

    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# 8. 实验记录
## model_addkp

### v 0.5
模型中加入 64 个 3DGS 采样点，每个 3DGS 采样 30 个点
主要修改部分：

    class FeatureExtractor(nn.Module)

next: **尝试引入点云法向量特征**

gpt建议：

1. 将法向量作为附加输入特征
将点云的法向量与点的坐标结合起来作为模型输入，法向量可以扩充点的几何信息，有助于模型学习局部表面细节。

input_features = torch.cat([points, normals], dim=-1)  # (B, N, 6)

2. 在特征提取模块中融合法向量
如果不想直接作为输入，可以在特征提取阶段引入法向量，将法向量作为局部几何描述，与点的坐标或其他特征进行融合。

方法：
将法向量与点的局部特征通过注意力机制、加权平均或 MLP 结合。
在特征提取模块中使用法向量指导特征的聚合或更新。


3. 设计法向量一致性损失
引入法向量损失约束补全点云的法向量与真实法向量一致，从而提高补全质量。

方法：
计算补全点云的法向量，与真实点云法向量对齐。
使用 余弦相似性损失 或 方向一致性损失

4. 使用法向量辅助点云补全（联合学习）
设计双任务网络，既预测完整点云的坐标，也预测补全点云的法向量。通过学习法向量，可以进一步提升点云的几何细节。
方法：
网络输出： 模型同时输出点云的坐标和法向量。
损失函数： 在位置损失（如 Chamfer Distance）基础上增加法向量一致性损失。


5. 点云和法向量的对比学习
使用对比学习方法，引导模型学习更细致的表面结构。可以构建点对间的法向量相似性，约束补全点云的局部几何。
