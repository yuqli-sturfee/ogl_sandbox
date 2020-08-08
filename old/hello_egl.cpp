//
// Created by yuqiong on 7/14/20.
//
// Include standard headers


#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <iostream>
#include <sstream>
#include <string>

static const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
};

static const int pbufferWidth = 9;
static const int pbufferHeight = 9;

static const EGLint pbufferAttribs[] = {
        EGL_WIDTH, pbufferWidth,
        EGL_HEIGHT, pbufferHeight,
        EGL_NONE,
};


void assertEGLError(const std::string& msg) {
    EGLint error = eglGetError();

    if (error != EGL_SUCCESS) {
        std::stringstream s;
        s << "EGL error 0x" << std::hex << error << " at " << msg;
        throw std::runtime_error(s.str());
    }
}


int main(int argc, char *argv[])
{
    // 1. Initialize EGL
//    EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    int deviceID = 0;  // TODO hardcode

    EGLDisplay eglDpy;
    EGLConfig config;
    EGLContext context;
    EGLint num_config;

    static const int MAX_DEVICES = 16;
    EGLDeviceEXT eglDevs[MAX_DEVICES];
    EGLint numDevices;

    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
            (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");

    eglQueryDevicesEXT(MAX_DEVICES, eglDevs, &numDevices);
    printf("Detected %d devices\n", numDevices);
    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
            (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");

    // Choose device by deviceID
    eglDpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, eglDevs[deviceID], nullptr);




    assertEGLError("eglGetDisplay");

    EGLint major, minor;

    eglInitialize(eglDpy, &major, &minor);

    // 2. Select an appropriate configuration
    EGLint numConfigs;
    EGLConfig eglCfg;

    eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);

    // 3. Create a surface
    EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg,
                                                 pbufferAttribs);

    // 4. Bind the API
    eglBindAPI(EGL_OPENGL_API);

    // 5. Create a context and make it current
    EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT,
                                         NULL);

    eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);

    // from now on use your OpenGL context

    // 6. Terminate EGL when finished
    eglTerminate(eglDpy);
    return 0;
}
