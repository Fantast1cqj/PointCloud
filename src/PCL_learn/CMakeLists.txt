cmake_minimum_required(VERSION 2.6 FATAL_ERROR)


# project(pcl_recognition_test)
# project(txt2pcd)

# project(pca)
project(kmeans)
# project(normals)
project(filter)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)


include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})


# add_executable(pcl_recognition_test src/correspondence_grouping.cpp)
# add_executable(txt2pcd src/txt2pcd.cpp)

# add_executable(pca pca/pca.cpp)
add_executable(kmeans kmeans/kmeans.cpp kmeans/kmeans_main.cpp)
# add_executable(normals utils/pcd_viewer.cpp pca/normals.cpp)
add_executable(filter filter/pass_through.cpp utils/pcd_viewer.cpp)


# target_link_libraries(pcl_recognition_test ${PCL_LIBRARIES})
# target_link_libraries(txt2pcd ${PCL_LIBRARIES})

# target_link_libraries(pca ${PCL_LIBRARIES})
target_link_libraries(kmeans ${PCL_LIBRARIES})
# target_link_libraries(normals ${PCL_LIBRARIES})
target_link_libraries(filter ${PCL_LIBRARIES})