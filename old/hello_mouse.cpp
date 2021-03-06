//
// Created by yuqiong on 7/14/20.
//
// Include standard headers

#include <iostream>
#include <string>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

GLFWwindow* window;

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "../tinyobjloader/tiny_obj_loader.h"

#include "shader.hpp"

#include <glm/gtc/type_ptr.hpp>

using namespace glm;

void loadMesh(std::string path, tinyobj::attrib_t &attrib, std::vector<tinyobj::shape_t> &shapes, std::vector<tinyobj::material_t> materials) {
    // Load custom data
    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str());
    std::cout << "LoadObj: return value " << ret << std::endl;

    if (!warn.empty()) {
        std::cout << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        exit(1);
    }
}


std::vector<float> singleMesh2Buffer(const tinyobj::shape_t & shape, const tinyobj::attrib_t & attrib) {
    std::cout << "combining a single mesh's face triangles to buffer data for OpenGL VBO..." << std::endl;
    std::vector<float> vertices_buffer;
    std::vector<float> colors_buffer;
    // Loop over faces(polygon)
    size_t index_offset = 0;
    for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
        int fv = shape.mesh.num_face_vertices[f];
        // Loop over vertices in the face.
        for (size_t v = 0; v < fv; v++) {
            // access to vertex
            tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
            tinyobj::real_t vx = attrib.vertices[3*idx.vertex_index+0];
            tinyobj::real_t vy = attrib.vertices[3*idx.vertex_index+1];
            tinyobj::real_t vz = attrib.vertices[3*idx.vertex_index+2];
            vertices_buffer.push_back(vx);
            vertices_buffer.push_back(vy);
            vertices_buffer.push_back(vz);
        }
        index_offset += fv;
    }

    return vertices_buffer;
}


std::vector<float> multipleMesh2Buffer(const std::vector<tinyobj::shape_t> & shapes, const tinyobj::attrib_t & attrib) {
    std::cout << "combining multiple meshs' face triangles to buffer data for OpenGL VBO..." << std::endl;
    std::cout << "shape size " << shapes.size() << std::endl;

    std::vector<float> buffer;  // store results

    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3*idx.vertex_index+0];
                tinyobj::real_t vy = attrib.vertices[3*idx.vertex_index+1];
                tinyobj::real_t vz = attrib.vertices[3*idx.vertex_index+2];
                // store results to buffer
                buffer.push_back(vx);
                buffer.push_back(vy);
                buffer.push_back(vz);
            }
            index_offset += fv;
        }
    }
    return buffer;
}

template <class T>
void normalizeVector(std::vector<T> & vec) {
    T val = 0;
    for (int i = 0; i < vec.size(); i++) {
        val = val > abs(vec[i]) ? val : abs(vec[i]);
    }

    for (int i = 0; i < vec.size(); i++) {
        vec[i] /= val ;
    }
}


// Initial position : on +Z
glm::vec3 position = glm::vec3( 0, 0, 5 );
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
    // load objs
    std::string inputfile = "../data/obj/cube.obj";
//    std::string inputfile = "../data/airboat.obj";
//    std::string inputfile = "../data/teddy.obj";
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    loadMesh(inputfile, attrib, shapes, materials);

    std::cout << "num of shapes ";
    std::cout << shapes.size() << std::endl;

    auto vertex_buffer_data = multipleMesh2Buffer(shapes, attrib);
    std::cout << "number of vertices " << vertex_buffer_data.size() << std::endl;

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
    glm::mat4 projection;

    // camera matrix
    glm::mat4 view;

    // model matrix : identity matrix, model at origin
    glm::mat4 model = glm::mat4(1.0f);

    // prepare mesh data
    GLuint vertex_buffer;
    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, vertex_buffer_data.size() * sizeof(vertex_buffer_data[0]), vertex_buffer_data.data(), GL_STATIC_DRAW);

    // read depth
    int x = 100;
    int y = 100;
    float z = 0;
    glReadPixels(x,y,1,1, GL_DEPTH_COMPONENT, GL_FLOAT, &z);
    std::cout << "Depth value of the selecetd pixel is " << z << std::endl;

    do {
        // clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // use the shader
        glUseProgram(programID);

        computeMatricesFromInputs(projection, view);

        glm::mat4 mvp = projection * view * model;

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
        glDrawArrays(GL_TRIANGLES, 0, 12*3);

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

