//
// Created by yuqiong on 7/14/20.
//
// Include standard headers

#include <iostream>
#include <string>
#include <vector>
#include <limits>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

GLFWwindow* window;

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include "shader.hpp"

#include "sturg_helper_func.hpp"
#include "sturg_loader.hpp"

using namespace glm;

void printVector(const std::vector<float> &vec, int start, int end) {
    for (int i = start; i < end; i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}


// Initial position : on +Z
glm::vec3 position = glm::vec3( 0, 0, 30 );
// Initial horizontal angle : toward -Z
float horizontalAngle = 3.14f;
// Initial vertical angle : none
float verticalAngle = 0.0f;
// Initial Field of View
float initialFoV = 45.0f;

float speed = 3.0f; // 3 units / second
float mouseSpeed = 0.005f;


void computeMatricesFromInputs(glm::mat4 & ProjectionMatrix, glm::mat4 & ViewMatrix){

    // glfwGetTime is called only once, the first time this function is called
    static double lastTime = glfwGetTime();

    // Compute time difference between current and last frame
    double currentTime = glfwGetTime();
    float deltaTime = float(currentTime - lastTime);

    // Get mouse position
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Reset mouse position for next frame
    glfwSetCursorPos(window, 1024/2, 768/2);

    // Compute new orientation
    horizontalAngle += mouseSpeed * float(1024/2 - xpos );
    verticalAngle   += mouseSpeed * float( 768/2 - ypos );

    // Direction : Spherical coordinates to Cartesian coordinates conversion
    glm::vec3 direction(
            cos(verticalAngle) * sin(horizontalAngle),
            sin(verticalAngle),
            cos(verticalAngle) * cos(horizontalAngle)
    );

    // Right vector
    glm::vec3 right = glm::vec3(
            sin(horizontalAngle - 3.14f/2.0f),
            0,
            cos(horizontalAngle - 3.14f/2.0f)
    );

    // Up vector
    glm::vec3 up = glm::cross( right, direction );

    // Move forward
    if (glfwGetKey( window, GLFW_KEY_UP ) == GLFW_PRESS){
        position += direction * deltaTime * speed;
    }
    // Move backward
    if (glfwGetKey( window, GLFW_KEY_DOWN ) == GLFW_PRESS){
        position -= direction * deltaTime * speed;
    }
    // Strafe right
    if (glfwGetKey( window, GLFW_KEY_RIGHT ) == GLFW_PRESS){
        position += right * deltaTime * speed;
    }
    // Strafe left
    if (glfwGetKey( window, GLFW_KEY_LEFT ) == GLFW_PRESS){
        position -= right * deltaTime * speed;
    }

    float FoV = initialFoV;// - 5 * glfwGetMouseWheel(); // Now GLFW 3 requires setting up a callback for this. It's a bit too complicated for this beginner's tutorial, so it's disabled instead.

    // Projection matrix : 45� Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
    ProjectionMatrix = glm::perspective(glm::radians(FoV), 4.0f / 3.0f, 0.1f, 100.0f);
    // Camera matrix
    ViewMatrix       = glm::lookAt(
            position,           // Camera is here
            position+direction, // and looks here : at the same position, plus "direction"
            up                  // Head is up (set to 0,-1,0 to look upside-down)
    );

    // For the next frame, the "last time" will be "now"
    lastTime = currentTime;
}

int main() {
    //////////////////////////////////////////////////////////////////////////////////////////
    // Initialize parameters
    //////////////////////////////////////////////////////////////////////////////////////////

    // load config file
    auto config_params = loadConfigFile("config.ini");

    // read command line params
    SturgInputParams input_params;
    input_params.csv_file_path = "/media/yuqiong/DATA/ogl_sandbox/assets/test_data/extern_param_single.csv";
    input_params.proj_matrix_csv_file = "/media/yuqiong/DATA/ogl_sandbox/assets/test_data/proj_matrix_portrait.csv";
    input_params.window_origins_file = "/media/yuqiong/DATA/ogl_sandbox/assets/test_data/windows.csv";
    input_params.center_x = 553096.755;
    input_params.center_y = 4183086.188;
    input_params.fov = 45;
    input_params.scene_width = 640;
    input_params.scene_height = 360;
    input_params.utm_prefix = "10N";
    input_params.radius = 500;
    input_params.image_width = 640;
    input_params.image_height = 360;

    // update input_params with config init data
    input_params.model_dir = config_params["geometry_models"];
    input_params.terrain_dir = config_params["geometry_terrain"];

    std::cout << "Using: " << std::endl;
    std::cout << " > Geometry Models Dir: " << input_params.model_dir << std::endl;
    std::cout << " > Geometry Terrain Dir: " << input_params.terrain_dir << std::endl;

    std::cout << " > Win Origins File: " << input_params.window_origins_file << std::endl;
    std::cout << " > utm prefix: " << input_params.utm_prefix << std::endl;

#if defined(CNN) && defined(CAFFE_OUT)
    std::vector<std::array<float, 2>> window_orig_params = searchParamData.getWindowOrigins();
#endif

    // loader
    SturgLoader sturg_loader;
    sturg_loader.init(input_params);
    sturg_loader.process();

    // get rendering data
    std::vector<GLfloat> vertices = sturg_loader.getVertices();
    std::vector<GLfloat> colors = sturg_loader.getColors();
    std::vector<GLuint> indices = sturg_loader.getIndices();

    std::cout << " >> vertices: " << vertices.size() << std::endl;
    std::cout << " >> colors: " << colors.size() << std::endl;
    std::cout << " >> indices: " << indices.size() << std::endl;

    std::cout << "Finished!\n";


    //////////////////////////////////////////////////////////////////////////////////////////
    // Initialize OpenGL
    //////////////////////////////////////////////////////////////////////////////////////////

    // initialize GLFW
    if (!glfwInit()) {
        std::cout << "Fail to initialize glfw" << std::endl;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // open a window and create its OpenGL context
    window = glfwCreateWindow(1024, 768, "Cube", nullptr, nullptr);
    if (!window) {
        std::cout << "Fail to create window!" << std::endl;
    }
    glfwMakeContextCurrent(window);

    // initialize glew
    glewExperimental = true;
    if (glewInit() != GLEW_OK) {
        std::cout  << "Fail to initialize GLEW" << std::endl;
    }

    // ensure we can capture the escape key
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // dark blue background
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

    // enable depth test
    glEnable(GL_DEPTH_TEST);
    // accept fragment if it's closer to the camera than the former one
    glDepthFunc(GL_LESS);

    glEnable(GL_CULL_FACE);

    // create VAO
    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // create and compile from the shaders
    GLuint programID = LoadShaders( "../src/SimpleVertexShader.vertexshader", "../src/SimpleFragmentShader.fragmentshader" );
    std::cout << "Program ID " << programID << std::endl;

    // get a handle for MVP transform
    GLuint MatrixID = glGetUniformLocation(programID, "MVP");

    // perspective matrix
    glm::mat4 projection ;

    // camera matrix
    glm::mat4 view ;

    // model matrix : identity matrix, model at origin
    glm::mat4 model = glm::mat4(1.0f);

    glm::mat4 mvp ;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // Prepare geometry and projection for rendering
    /////////////////////////////////////////////////////////////////////////////////////////////
    // Load vertices into a VBO
    GLuint vertex_buffer = 0;  // Vertex buffers to store vertices
    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), &vertices[0],
                 GL_STATIC_DRAW);

    // Load colors into a VBO
    GLuint color_buffer = 0;   // Color buffers to store RGB
    glGenBuffers(1, &color_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(GLfloat), &colors[0], GL_STATIC_DRAW);

    /////////////////////////////////////////////////////////////////////////////////////////////
    // Actual rendering loop
    /////////////////////////////////////////////////////////////////////////////////////////////
    do {
        // clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // use the shader
        glUseProgram(programID);

        // get model view projection from the input
        computeMatricesFromInputs(projection, view);
        mvp = projection * view * model;

        // send transforms
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]);

        // 1st attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
        glVertexAttribPointer(
                0,
                3,
                GL_FLOAT,
                GL_FALSE,
                0,
                (void*) 0
                );

        // draw the geometry!
        glDrawArrays(GL_TRIANGLES, 0, vertices.size());

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);

        // swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
    }  // check if the esc key has been pressed or window closed
    while ( glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

    // clean up VBO
    glDeleteBuffers(1, &vertex_buffer);
    glDeleteVertexArrays(1, &VertexArrayID);
    glDeleteProgram(programID);

    // close OpenGL window and terminate GLFW
    glfwTerminate();


}

