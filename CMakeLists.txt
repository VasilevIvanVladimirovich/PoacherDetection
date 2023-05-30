cmake_minimum_required(VERSION 3.5)

project(PoacherDetection VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        mainwindow.h
        VideoThread.cpp
        VideoThread.h
        CustomGraphicsView.cpp
        CustomGraphicsView.h
        HumanDetector.cpp
        HumanDetector.h
        Defs.h
)

SET(OpenCV_DIR "C:/Work/extern/pre-opencv/build")
SET(OpenCV_LIB "C:/Work/extern/pre-opencv/build/x64/vc15/lib/opencv_world455.lib")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIR}/include )
link_libraries(${OpenCV_LIB})



#link_libraries(
#    opencv_core452
#    )

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(PoacherDetection
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET PoacherDetection APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(PoacherDetection SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(PoacherDetection
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(PoacherDetection PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)



set_target_properties(PoacherDetection PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

install(TARGETS PoacherDetection
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(PoacherDetection)
endif()



