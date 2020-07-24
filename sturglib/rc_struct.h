//
//  rc_struct.h
//  rc structs
//
//  Created by Dilip Patlolla on 09/21/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#ifndef RC_STRUCT_H
#define RC_STRUCT_H

#include <stdio.h>
#include <string.h>
#include <algorithm>  // std::max
#include <cmath>      // std::ceil
#include <iostream>   // std::std::cout
#include <iterator>   // std:: distance
#include <list>       // std::list<>
#include <vector>     // std::vector<>

// Include CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <helper_math.h>

// Include GLM
#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <glm/gtx/quaternion.hpp>

struct MbrRectangles {
    std::vector<double> id;
    std::vector<float3> min_vertices;
    std::vector<float3> max_vertices;
};

struct MbrLsForSF {
    std::vector<float> min_vertex_x;
    std::vector<float> max_vertex_x;
    std::vector<float> min_vertex_y;
    std::vector<float> max_vertex_y;
};

struct MbrFaces {
    std::vector<double> id;
    std::vector<float3> x_face, y_face, z_face;
    std::vector<float3> mbrFace_sub;
};

struct MbrRectangle {
    double id;
    float3 min_vertices;
    float3 max_vertices;
};

struct RayHitData {
    unsigned int flag;
    double id;
    unsigned int index;
    float t_min;
    double3 loc;
    double2 pt;
};

/// The position and orientation of a camera at a point in time.
struct CameraPose {
   public:
    glm::quat orientation;
    glm::dvec3 position;

    CameraPose(){};
    CameraPose(glm::quat orientation, glm::dvec3 position)
        : orientation(orientation), position(position){};
};

// Ray structure
struct Ray {
    __device__ Ray(){};
    __device__ Ray(double3 &o, double3 &d) {
        orig = o;
        dest = d;

        dir = make_double3(dest.x - orig.x, dest.y - orig.y, dest.z - orig.z);
        // dir = normalize(dir);
        inv_dir = make_double3(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
    }
    double2 pt;
    double3 orig, dest;
    double3 dir;
    double3 inv_dir;
};

#endif /* RC_STRUCT_H */
