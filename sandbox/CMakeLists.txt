#cmake_minimum_required (VERSION 3.18.1)
cmake_minimum_required (VERSION 3.10.1)
project (sandbox LANGUAGES CXX CUDA)
#project (sandbox LANGUAGES CXX)

# add OpenGL

#find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL GLX)
#include_directories(${OPENGL_INCLUDE_DIRS})
#if(OPENGL_FOUND)
#    message("Found OpenGL in the current environment!")
#else()
#    message("Error: No OpenGL found.")
#endif()
#
#message("OpenGL include dirs" )
#message("${OPENGL_INCLUDE_DIR}")
#message("EGL include dirs" )
#message("${OPENGL_EGL_INCLUDE_DIRS}")
#
#if (OpenGL_EGL_FOUND)
#    message("EGL Found!")
#else()
#    message("EGL Not Found!")
#endif()
#
#message(${OPENGL_egl_LIBRARY})
#message(${OPENGL_glx_LIBRARY})
#message(${OPENGL_opengl_LIBRARY})
#
## Add GLFW
#find_package(glfw3 REQUIRED)
#
## Add GLEW
#find_package(GLEW REQUIRED)
#

find_package(OpenCV REQUIRED)

#find_package(CUDA REQUIRED)

add_executable (sandbox postprocess.cu)
#add_executable (sandbox "hello_headless.cpp")
target_link_libraries(sandbox PRIVATE ${OpenCV_LIBS} )

#target_link_libraries(sandbox PRIVATE glfw GLEW::GLEW ${OpenCV_LIBS} OpenGL::OpenGL OpenGL::EGL OpenGL::GLX)

set_target_properties(sandbox PROPERTIES CXX_STANDARD 11)




