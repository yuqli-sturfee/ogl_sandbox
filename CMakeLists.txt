cmake_minimum_required (VERSION 3.5.1)
project (sandbox LANGUAGES CXX CUDA)

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

include_directories("${CMAKE_SOURCE_DIR}/sturglib")
#include_directories("${CMAKE_SOURCE_DIR}/io")

file(GLOB STURG_SOURCES
        "sturglib/*.h*"
        "sturglib/*.cpp"
        "sturglib/*.cu"
        )

#add_executable (sandbox "src/render.cpp" ${STURG_SOURCES} "src/shader.cpp")
add_executable (sandbox "src/sandbox_cam_search.cpp" ${STURG_SOURCES} )

#target_link_libraries(sandbox ${OPENGL_LIBRARIES} glfw  GLEW::GLEW )
set_target_properties(sandbox PROPERTIES CXX_STANDARD 11)



