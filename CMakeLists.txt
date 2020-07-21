cmake_minimum_required (VERSION 3.5.1)
project (sandbox)

# add OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})
if(OPENGL_FOUND)
    message("Found OpenGL in the current environment!")
else()
    message("Error: No OpenGL found.")
endif()

# Add GLFW
find_package(glfw3 REQUIRED)

# Add GLEW
find_package(GLEW REQUIRED)

# add glm
find_package(glm REQUIRED)

# Add objloader
add_subdirectory(tinyobjloader)

add_executable (sandbox src/render.cpp src/shader.hpp src/shader.cpp)

target_link_libraries(sandbox ${OPENGL_LIBRARIES} glfw  GLEW::GLEW glm)

