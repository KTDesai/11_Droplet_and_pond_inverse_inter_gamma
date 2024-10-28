cmake_minimum_required(VERSION 3.11...3.28)

project(SIMPLE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

#add_executable(main main.cc)
add_executable(main main.cc point.cc vector.cc face.cc cell.cc boundary.cc mesh.cc scalar_field.cc vector_field.cc pv_coupling.cc scalar_boundary_field.cc vector_boundary_field.cc)
set(Eigen3_DIR "/usr/local/lib/cmake/eigen3")
find_package(Eigen3 REQUIRED)
target_link_libraries(main PUBLIC Eigen3::Eigen)

 find_package(OpenMP)
 if(OpenMP_CXX_FOUND)
 target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
 endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json
)
