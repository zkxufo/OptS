# Install script for directory: /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/include/pybind11")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/pybind11" TYPE FILE FILES
    "/home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build/tests/test_cmake_build/subdirectory_embed/pybind11/pybind11Config.cmake"
    "/home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build/tests/test_cmake_build/subdirectory_embed/pybind11/pybind11ConfigVersion.cmake"
    "/home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tools/FindPythonLibsNew.cmake"
    "/home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tools/pybind11Common.cmake"
    "/home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tools/pybind11Tools.cmake"
    "/home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tools/pybind11NewTools.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/pybind11/test_export.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/pybind11/test_export.cmake"
         "/home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build/tests/test_cmake_build/subdirectory_embed/pybind11/CMakeFiles/Export/share/cmake/pybind11/test_export.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/pybind11/test_export-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/pybind11/test_export.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/pybind11" TYPE FILE FILES "/home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build/tests/test_cmake_build/subdirectory_embed/pybind11/CMakeFiles/Export/share/cmake/pybind11/test_export.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/pybind11" TYPE FILE FILES "/home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build/tests/test_cmake_build/subdirectory_embed/pybind11/CMakeFiles/Export/share/cmake/pybind11/test_export-release.cmake")
  endif()
endif()

