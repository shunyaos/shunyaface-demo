cmake_minimum_required(VERSION 3.8)
add_definitions(-DCPU_ONLY)
SET(SRC
    	# Give path of the cpp file here 
        main.cpp
)

SET(INC_DIR /usr)

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
find_package( OpenCV REQUIRED)

set(CMAKE_BUILD_TYPE release)
set(CMAKE_CXX_STANDARD 11)
add_definitions(-fPIC)
add_definitions(-Ofast)
add_definitions(-ffast-math)

include_directories(${OpenCV_DIR}/include)
include_directories(/usr/include/fr/)
include_directories(/usr/include/fr/models/)
include_directories(${INC_DIR}/include)
add_executable(facedemo ${SRC})
TARGET_LINK_LIBRARIES(facedemo ${OpenCV_LIBS} -lboost_system -lshunyaface)
