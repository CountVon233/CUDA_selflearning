# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/countvon/CUDA_Project/matrix_mul

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/countvon/CUDA_Project/matrix_mul/build

# Include any dependencies generated for this target.
include CMakeFiles/matrix_mul.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/matrix_mul.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/matrix_mul.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/matrix_mul.dir/flags.make

CMakeFiles/matrix_mul.dir/main.cpp.o: CMakeFiles/matrix_mul.dir/flags.make
CMakeFiles/matrix_mul.dir/main.cpp.o: ../main.cpp
CMakeFiles/matrix_mul.dir/main.cpp.o: CMakeFiles/matrix_mul.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/countvon/CUDA_Project/matrix_mul/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/matrix_mul.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/matrix_mul.dir/main.cpp.o -MF CMakeFiles/matrix_mul.dir/main.cpp.o.d -o CMakeFiles/matrix_mul.dir/main.cpp.o -c /home/countvon/CUDA_Project/matrix_mul/main.cpp

CMakeFiles/matrix_mul.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matrix_mul.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/countvon/CUDA_Project/matrix_mul/main.cpp > CMakeFiles/matrix_mul.dir/main.cpp.i

CMakeFiles/matrix_mul.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matrix_mul.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/countvon/CUDA_Project/matrix_mul/main.cpp -o CMakeFiles/matrix_mul.dir/main.cpp.s

CMakeFiles/matrix_mul.dir/matrix_mul.cu.o: CMakeFiles/matrix_mul.dir/flags.make
CMakeFiles/matrix_mul.dir/matrix_mul.cu.o: ../matrix_mul.cu
CMakeFiles/matrix_mul.dir/matrix_mul.cu.o: CMakeFiles/matrix_mul.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/countvon/CUDA_Project/matrix_mul/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/matrix_mul.dir/matrix_mul.cu.o"
	/usr/local/cuda-12.2/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/matrix_mul.dir/matrix_mul.cu.o -MF CMakeFiles/matrix_mul.dir/matrix_mul.cu.o.d -x cu -c /home/countvon/CUDA_Project/matrix_mul/matrix_mul.cu -o CMakeFiles/matrix_mul.dir/matrix_mul.cu.o

CMakeFiles/matrix_mul.dir/matrix_mul.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/matrix_mul.dir/matrix_mul.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/matrix_mul.dir/matrix_mul.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/matrix_mul.dir/matrix_mul.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target matrix_mul
matrix_mul_OBJECTS = \
"CMakeFiles/matrix_mul.dir/main.cpp.o" \
"CMakeFiles/matrix_mul.dir/matrix_mul.cu.o"

# External object files for target matrix_mul
matrix_mul_EXTERNAL_OBJECTS =

matrix_mul: CMakeFiles/matrix_mul.dir/main.cpp.o
matrix_mul: CMakeFiles/matrix_mul.dir/matrix_mul.cu.o
matrix_mul: CMakeFiles/matrix_mul.dir/build.make
matrix_mul: CMakeFiles/matrix_mul.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/countvon/CUDA_Project/matrix_mul/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable matrix_mul"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matrix_mul.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matrix_mul.dir/build: matrix_mul
.PHONY : CMakeFiles/matrix_mul.dir/build

CMakeFiles/matrix_mul.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/matrix_mul.dir/cmake_clean.cmake
.PHONY : CMakeFiles/matrix_mul.dir/clean

CMakeFiles/matrix_mul.dir/depend:
	cd /home/countvon/CUDA_Project/matrix_mul/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/countvon/CUDA_Project/matrix_mul /home/countvon/CUDA_Project/matrix_mul /home/countvon/CUDA_Project/matrix_mul/build /home/countvon/CUDA_Project/matrix_mul/build /home/countvon/CUDA_Project/matrix_mul/build/CMakeFiles/matrix_mul.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/matrix_mul.dir/depend

