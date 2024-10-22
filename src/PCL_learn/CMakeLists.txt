cmake_minimum_required(VERSION 2.6 FATAL_ERROR)


# project(pcl_recognition_test)
# project(txt2pcd)
# project(pca)
# project(kmeans)
# project(normals)
project(filter)
project(voxelgrid)
project(indices)
project(denoise)
project(RANSAC_line)
project(RANSAC_lines)
project(RANSAC_plane)
project(RANSAC_ball)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)


include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})


# add_executable(pcl_recognition_test src/correspondence_grouping.cpp)
# add_executable(txt2pcd src/txt2pcd.cpp)

# add_executable(pca pca/pca.cpp)
# add_executable(kmeans kmeans/kmeans.cpp kmeans/kmeans_main.cpp)
# add_executable(normals utils/pcd_viewer.cpp pca/normals.cpp)
add_executable(filter filter/pass_through.cpp utils/pcd_viewer.cpp)
add_executable(voxelgrid filter/voxel_grid.cpp utils/pcd_viewer.cpp)
add_executable(indices filter/indices.cpp utils/pcd_viewer.cpp)
add_executable(denoise filter/denoise.cpp utils/pcd_viewer.cpp)
add_executable(RANSAC_line RANSAC/line.cpp utils/pcd_viewer.cpp)
add_executable(RANSAC_lines RANSAC/multi_line.cpp utils/pcd_viewer.cpp)
add_executable(RANSAC_plane RANSAC/plane.cpp utils/pcd_viewer.cpp)
add_executable(RANSAC_ball RANSAC/ball.cpp utils/pcd_viewer.cpp)



# target_link_libraries(pcl_recognition_test ${PCL_LIBRARIES})
# target_link_libraries(txt2pcd ${PCL_LIBRARIES})

# target_link_libraries(pca ${PCL_LIBRARIES})
# target_link_libraries(kmeans ${PCL_LIBRARIES})
# target_link_libraries(normals ${PCL_LIBRARIES})
target_link_libraries(filter ${PCL_LIBRARIES})
target_link_libraries(voxelgrid ${PCL_LIBRARIES})
target_link_libraries(indices ${PCL_LIBRARIES})
target_link_libraries(denoise ${PCL_LIBRARIES})
target_link_libraries(RANSAC_line ${PCL_LIBRARIES})
target_link_libraries(RANSAC_lines ${PCL_LIBRARIES})
target_link_libraries(RANSAC_plane ${PCL_LIBRARIES})
target_link_libraries(RANSAC_ball ${PCL_LIBRARIES})