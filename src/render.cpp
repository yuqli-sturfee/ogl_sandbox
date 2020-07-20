//
// Created by yuqiong on 7/14/20.
//
// Include standard headers

#include <iostream>
#include <string>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>


// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

// Include GLM
#include <glm/glm.hpp>

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "../tinyobjloader/tiny_obj_loader.h"

#include "shader.hpp"

using namespace glm;

void loadMesh(std::string path, tinyobj::attrib_t &attrib, std::vector<tinyobj::shape_t> &shapes, std::vector<tinyobj::material_t> materials) {
    // Load custom data
    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str());
    std::cout << "return value " << ret << std::endl;

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


void inspectMesh(const std::vector<tinyobj::shape_t> & shapes, const tinyobj::attrib_t & attrib) {
    // Loop over shapes
    std::cout << "Shape size " << shapes.size() << std::endl;
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
                tinyobj::real_t nx = attrib.normals[3*idx.normal_index+0];
                tinyobj::real_t ny = attrib.normals[3*idx.normal_index+1];
                tinyobj::real_t nz = attrib.normals[3*idx.normal_index+2];
                tinyobj::real_t tx = attrib.texcoords[2*idx.texcoord_index+0];
                tinyobj::real_t ty = attrib.texcoords[2*idx.texcoord_index+1];

                // Optional: vertex colors
                // tinyobj::real_t red = attrib.colors[3*idx.vertex_index+0];
                // tinyobj::real_t green = attrib.colors[3*idx.vertex_index+1];
                // tinyobj::real_t blue = attrib.colors[3*idx.vertex_index+2];
            }
            index_offset += fv;

            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }
}


std::vector<std::vector<float>> singleMesh2Buffer(const tinyobj::shape_t & shape, const tinyobj::attrib_t & attrib) {
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

            tinyobj::real_t red = attrib.colors[3*idx.vertex_index+0];
            tinyobj::real_t green = attrib.colors[3*idx.vertex_index+1];
            tinyobj::real_t blue = attrib.colors[3*idx.vertex_index+2];
            colors_buffer.push_back(red);
            colors_buffer.push_back(green);
            colors_buffer.push_back(blue);
        }
        index_offset += fv;
    }

    return {vertices_buffer, colors_buffer};
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


int main() {
    // load objs
    std::string inputfile = "../data/cube.obj";
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    loadMesh(inputfile, attrib, shapes, materials);

    std::cout << "num of shapes ";
    std::cout << shapes.size() << std::endl;

    auto buffers = singleMesh2Buffer(shapes[0], attrib);
    std::vector<float> vertex_buffer_data = buffers[0];
    std::vector<float> color_buffer_data = buffers[1];

//    // should have 12 * 3 * 3 = 108 data points : 12 traingles * 3 points * xyz
//    std::cout << vertex_buffer_data.size() << std::endl;
//    for (int i = 0; i < vertex_buffer_data.size(); i++) {
//        if (i % 3 == 0 && i != 0) {
//            std::cout << std::endl;
//        }
//        std::cout << vertex_buffer_data[i] << " ";
//    }



    // initialize GLFW
    if (!glfwInit()) {
        std::cout << "Fail to initialize glfw" << std::endl;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // open a window and create its OpenGL context
    window = glfwCreateWindow(1024, 768, "Triangle", nullptr, nullptr);
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

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // create and compile from the shaders
    GLuint programID = LoadShaders( "../src/SimpleVertexShader.vertexshader", "../src/SimpleFragmentShader.fragmentshader" );
    std::cout << "Program ID " << programID << std::endl;

    GLuint vertex_buffer;
    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_data.data()), vertex_buffer_data.data(), GL_STATIC_DRAW);

//    GLuint color_buffer;
//    glGenBuffers(1, &color_buffer);
//    glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
//    glBufferData(GL_ARRAY_BUFFER, sizeof(color_buffer_data.data()), color_buffer_data.data(), GL_STATIC_DRAW);

    do {
        // clear the screen
        glClear(GL_COLOR_BUFFER_BIT);

        // use the shader
        glUseProgram(programID);

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

        // draw the triangle!
        glDrawArrays(GL_TRIANGLES, 0, 12*3);

        glDisableVertexAttribArray(0);

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

