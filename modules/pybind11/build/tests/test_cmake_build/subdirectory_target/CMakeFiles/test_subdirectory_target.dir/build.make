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
CMAKE_SOURCE_DIR = /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/subdirectory_target

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build/tests/test_cmake_build/subdirectory_target

# Include any dependencies generated for this target.
include CMakeFiles/test_subdirectory_target.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_subdirectory_target.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_subdirectory_target.dir/flags.make

CMakeFiles/test_subdirectory_target.dir/home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp.o: CMakeFiles/test_subdirectory_target.dir/flags.make
CMakeFiles/test_subdirectory_target.dir/home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp.o: /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build/tests/test_cmake_build/subdirectory_target/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_subdirectory_target.dir/home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp.o"
	/home/linuxbrew/.linuxbrew/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_subdirectory_target.dir/home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp.o -c /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp

CMakeFiles/test_subdirectory_target.dir/home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_subdirectory_target.dir/home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp.i"
	/home/linuxbrew/.linuxbrew/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp > CMakeFiles/test_subdirectory_target.dir/home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp.i

CMakeFiles/test_subdirectory_target.dir/home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_subdirectory_target.dir/home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp.s"
	/home/linuxbrew/.linuxbrew/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp -o CMakeFiles/test_subdirectory_target.dir/home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp.s

# Object files for target test_subdirectory_target
test_subdirectory_target_OBJECTS = \
"CMakeFiles/test_subdirectory_target.dir/home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp.o"

# External object files for target test_subdirectory_target
test_subdirectory_target_EXTERNAL_OBJECTS =

test_cmake_build.cpython-39-x86_64-linux-gnu.so: CMakeFiles/test_subdirectory_target.dir/home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/main.cpp.o
test_cmake_build.cpython-39-x86_64-linux-gnu.so: CMakeFiles/test_subdirectory_target.dir/build.make
test_cmake_build.cpython-39-x86_64-linux-gnu.so: CMakeFiles/test_subdirectory_target.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build/tests/test_cmake_build/subdirectory_target/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module test_cmake_build.cpython-39-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_subdirectory_target.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_subdirectory_target.dir/build: test_cmake_build.cpython-39-x86_64-linux-gnu.so

.PHONY : CMakeFiles/test_subdirectory_target.dir/build

CMakeFiles/test_subdirectory_target.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_subdirectory_target.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_subdirectory_target.dir/clean

CMakeFiles/test_subdirectory_target.dir/depend:
	cd /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build/tests/test_cmake_build/subdirectory_target && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/subdirectory_target /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/tests/test_cmake_build/subdirectory_target /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build/tests/test_cmake_build/subdirectory_target /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build/tests/test_cmake_build/subdirectory_target /home/h2amer/work/workspace/JPEG_HDQ/DNN_HDQ/pybind11/build/tests/test_cmake_build/subdirectory_target/CMakeFiles/test_subdirectory_target.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_subdirectory_target.dir/depend

