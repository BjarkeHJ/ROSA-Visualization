cmake_minimum_required(VERSION 3.8)
project(rosa_dev)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(EXEC_NAME rosa_dev)

# set(CMAKE_INSTALL_RPATH "/home/airlab/miniconda3/envs/isaac_env/lib")
# set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

find_package(PCL 1.12 REQUIRED)

include_directories(include)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(${EXEC_NAME}
               src/rosa_exec.cpp
               src/rosa_main.cpp
               src/Extra_Del.cpp)

target_include_directories(${EXEC_NAME} PRIVATE ${PCL_INCLUDE_DIRS})
target_include_directories(${EXEC_NAME} PRIVATE include)
target_link_libraries(${EXEC_NAME} ${PCL_LIBRARIES})


# install(TARGETS ${EXEC_NAME} DESTINATION lib/${PROJECT_NAME})

