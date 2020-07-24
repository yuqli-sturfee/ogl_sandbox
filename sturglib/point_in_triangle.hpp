//
//  point_in_triangle.hpp
//  spatial filtering
//
//  Created by Dilip Patlolla on 11/02/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#ifndef POINT_IN_TRIANGLE_HPP
#define POINT_IN_TRIANGLE_HPP

#include "rc_struct.h"
#include "sturg_struct.h"

#define NO_HIT -9999

float point_in_triangle(size_t i, double x, double y, std::vector<SturGVertex> vertices);
float getHeight(float2 v0, float2 v1, float2 v2, float3 vector_of_zs);

#endif /* POINT_IN_TRIANGLE_HPP */
