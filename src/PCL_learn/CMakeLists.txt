cmake_minimum_required(VERSION 2.6 FATAL_ERROR)


project(pcl_recognition_test)
project(txt2pcd)
# project(pcd_viewer)
project(pca)
project(normals)
project(kmeans)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)


include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})


add_executable(pcl_recognition_test src/correspondence_grouping.cpp)
add_executable(txt2pcd src/txt2pcd.cpp)
# add_executable(pcd_viewer src/pcd_viewer.cpp)
add_executable(pca pca/pca.cpp)
add_executable(kmeans kmeans/kmeans.cpp kmeans/kmeans_main.cpp)
add_executable(normals src/pcd_viewer.cpp pca/normals.cpp)


target_link_libraries(pcl_recognition_test ${PCL_LIBRARIES})
target_link_libraries(txt2pcd ${PCL_LIBRARIES})
# target_link_libraries(pcd_viewer ${PCL_LIBRARIES})
target_link_libraries(pca ${PCL_LIBRARIES})
target_link_libraries(kmeans ${PCL_LIBRARIES})
target_link_libraries(normals ${PCL_LIBRARIES})