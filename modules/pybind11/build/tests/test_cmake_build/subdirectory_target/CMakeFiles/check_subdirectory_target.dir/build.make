# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_cmake_build/subdirectory_target

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build/tests/test_cmake_build/subdirectory_target

# Utility rule file for check_subdirectory_target.

# Include the progress variables for this target.
include CMakeFiles/check_subdirectory_target.dir/progress.make

CMakeFiles/check_subdirectory_target: test_cmake_build.cpython-39-x86_64-linux-gnu.so
	/usr/bin/cmake -E env PYTHONPATH=/home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build/tests/test_cmake_build/subdirectory_target /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ_env/bin/python /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_cmake_build/subdirectory_target/../test.py test_subdirectory_target

check_subdirectory_target: CMakeFiles/check_subdirectory_target
check_subdirectory_target: CMakeFiles/check_subdirectory_target.dir/build.make

.PHONY : check_subdirectory_target

# Rule to build all files generated by this target.
CMakeFiles/check_subdirectory_target.dir/build: check_subdirectory_target

.PHONY : CMakeFiles/check_subdirectory_target.dir/build

CMakeFiles/check_subdirectory_target.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/check_subdirectory_target.dir/cmake_clean.cmake
.PHONY : CMakeFiles/check_subdirectory_target.dir/clean

CMakeFiles/check_subdirectory_target.dir/depend:
	cd /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build/tests/test_cmake_build/subdirectory_target && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_cmake_build/subdirectory_target /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_cmake_build/subdirectory_target /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build/tests/test_cmake_build/subdirectory_target /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build/tests/test_cmake_build/subdirectory_target /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build/tests/test_cmake_build/subdirectory_target/CMakeFiles/check_subdirectory_target.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/check_subdirectory_target.dir/depend
