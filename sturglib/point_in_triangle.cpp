//
//  point_in_triangle.cpp
//  spatial filtering
//
//  Created by Dilip Patlolla on 11/02/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#include "point_in_triangle.hpp"

float point_in_triangle(size_t i, double x, double y, std::vector<SturGVertex> vertices) {
    float2 A = make_float2(vertices[i * 3].vertex_param[0], vertices[i * 3].vertex_param[1]);
    float2 B =
        make_float2(vertices[i * 3 + 1].vertex_param[0], vertices[i * 3 + 1].vertex_param[1]);
    float2 C =
        make_float2(vertices[i * 3 + 2].vertex_param[0], vertices[i * 3 + 2].vertex_param[1]);
    float2 P = make_float2(x, y);

    float3 vector_of_zs =
        make_float3(vertices[i * 3].vertex_param[2], vertices[i * 3 + 1].vertex_param[2],
                    vertices[i * 3 + 2].vertex_param[2]);

    float2 v0 = C - A;
    float2 v1 = B - A;
    float2 v2 = P - A;

    float height = getHeight(v0, v1, v2, vector_of_zs);

    return height;
}

float getHeight(float2 v0, float2 v1, float2 v2, float3 vector_of_zs) {
    float height = NO_HIT;
    float dot_0_0 = dot(v0, v0);
    float dot_0_1 = dot(v0, v1);
    float dot_0_2 = dot(v0, v2);
    float dot_1_1 = dot(v1, v1);
    float dot_1_2 = dot(v1, v2);
    float denom = (dot_0_0 * dot_1_1 - dot_0_1 * dot_0_1);
    if (denom == 0) {
        return 0;
    }

    float inv_denom = 1.0f / denom;

    float u = (dot_1_1 * dot_0_2 - dot_0_1 * dot_1_2) * inv_denom;
    float v = (dot_0_0 * dot_1_2 - dot_0_1 * dot_0_2) * inv_denom;

    // std::cout << "U,V" << u << "," << v << std::endl;

    if ((u >= 0.0f) && (v >= 0.0f) && ((u + v) < 1.0f)) {
        height = (1 - u - v) * vector_of_zs.x + v * vector_of_zs.y + u * vector_of_zs.z;
        // std::cout << "height here: " << height << std::endl;
        // return height;
    } else {
        height = NO_HIT;
        // return height;
    }

    return height;
}
