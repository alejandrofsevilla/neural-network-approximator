cmake_minimum_required(VERSION 3.22.0)

set(OPENNN_VERSION "v6.0.3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(neural-network-approximator LANGUAGES CXX)

include(FetchContent)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
fetchContent_Declare(opennn
  GIT_REPOSITORY https://github.com/Artelnics/opennn.git
  GIT_TAG ${OPENNN_VERSION})
fetchContent_MakeAvailable(opennn)

add_executable(${PROJECT_NAME} Main.cpp)
target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wextra -Wpedantic -O0)
target_include_directories(${PROJECT_NAME} PRIVATE ${opennn_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${boost_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE opennn Boost::program_options)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

