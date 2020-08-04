cmake_minimum_required (VERSION 3.5.1)
#project (sandbox LANGUAGES CXX CUDA)
project (sandbox LANGUAGES CXX)

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

find_package(OpenCV REQUIRED)

#find_package(CUDA REQUIRED)

add_executable (sandbox "hello.cpp" "shader.cpp")
#add_executable (sandbox "hello-no-fbo.cpp" "shader.cpp")
#add_executable (sandbox "hello-pbo.cpp" "shader.cpp")

target_link_libraries(sandbox glfw GLEW::GLEW ${OPENGL_LIBRARIES} ${OpenCV_LIBS})

#target_include_directories(sandbox PRIVATE "/usr/include/EGL")
#target_link_libraries(sandbox "/usr/lib/x86_64-linux-gnu/libEGL.so")

set_target_properties(sandbox PROPERTIES CXX_STANDARD 11)



