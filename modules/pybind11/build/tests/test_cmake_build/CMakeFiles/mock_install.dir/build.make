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
CMAKE_SOURCE_DIR = /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build

# Utility rule file for mock_install.

# Include the progress variables for this target.
include tests/test_cmake_build/CMakeFiles/mock_install.dir/progress.make

tests/test_cmake_build/CMakeFiles/mock_install:
	cd /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build/tests/test_cmake_build && /usr/bin/cmake -DCMAKE_INSTALL_PREFIX=/home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build/mock_install -P /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build/cmake_install.cmake

mock_install: tests/test_cmake_build/CMakeFiles/mock_install
mock_install: tests/test_cmake_build/CMakeFiles/mock_install.dir/build.make

.PHONY : mock_install

# Rule to build all files generated by this target.
tests/test_cmake_build/CMakeFiles/mock_install.dir/build: mock_install

.PHONY : tests/test_cmake_build/CMakeFiles/mock_install.dir/build

tests/test_cmake_build/CMakeFiles/mock_install.dir/clean:
	cd /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build/tests/test_cmake_build && $(CMAKE_COMMAND) -P CMakeFiles/mock_install.dir/cmake_clean.cmake
.PHONY : tests/test_cmake_build/CMakeFiles/mock_install.dir/clean

tests/test_cmake_build/CMakeFiles/mock_install.dir/depend:
	cd /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11 /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build/tests/test_cmake_build /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build/tests/test_cmake_build/CMakeFiles/mock_install.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/test_cmake_build/CMakeFiles/mock_install.dir/depend

