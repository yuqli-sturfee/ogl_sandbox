//
//  strug_struct.h
//  sturgStruct
//
//  Created by Dilip Patlolla on 1/24/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#ifndef sturg_struct_h
#define sturg_struct_h

#include <string.h>
#include <algorithm>  // std::max
#include <cmath>      // std::ceil
#include <iostream>   // std::std::cout
#include <iterator>   // std:: distance
#include <list>       // std::list<>
#include <vector>     // std::vector<>

#define TRUE 1
#define FALSE 0

#define NUM_OF_BANDS 1
#define COLOR_FMT GL_RGBA
#define COLOR_FMT_SIZE 16
#define NUM_OF_CNN_INPUT_BANDS 6
#define NUM_OF_OPENGL_OUTPUT_BANDS 4

// model data defines
#define FACE_VERTICES_SIZE 3
#define VERTEX_VERTICES_SIZE 3
#define COLOR_PARAM_SIZE 3

// constant tile size
#define TILE_SIZE 300

// using namespace std;
struct SturgInputParams {
    bool verbose = FALSE;
    bool write_output = false;
    bool use_edge = false;

    int scene_width = 0;
    int scene_height = 0;
    int image_width = 0;
    int image_height = 0;
    float radius = 0;
    float fov = 0;
    double center_x = 0;
    double center_y = 0;
    double cam_height = 0;

    std::string utm_prefix;
    std::string image_ref_edge;
    std::string window_origins_file;
    std::string csv_file_path;
    std::string output_dir;
    std::string proj_matrix_csv_file;
    std::string pca_mean_file_path;
    std::string pca_coeff_file_path;
    std::string model_dir;
    std::string terrain_dir;
    std::string caffe_train_file;
    std::string caffe_model_file;
};

struct SturGTileMetaData {
    uint32_t version, meta_length;
    uint32_t models_count;
    uint32_t tile_center_x, tile_center_y, tile_center_z;
};

struct SturGVertex {
    float vertex_param[VERTEX_VERTICES_SIZE] = {0, 0, 0};
    bool operator<(const SturGVertex that) const {
        return memcmp((void*)this, (void*)&that, sizeof(SturGVertex)) > 0;
    };
};

struct SturGFace {
    uint16_t face_vertex[FACE_VERTICES_SIZE] = {0, 0, 0};
};

struct SturGBuildingData {
    double id;
    uint32_t vertices_byte_length;
    uint32_t faces_byte_length;
    uint32_t count_vertices, count_faces;
    std::vector<SturGVertex> vertices;
    std::vector<SturGFace> faces;
    bool is_uint_16;
    unsigned int is_terrain;
    size_t vertices_data_type_size;
};

struct SturgCameraParameters {
    std::string param_name;
    double cam_x, cam_y, cam_z;
    float yaw, pitch, roll;
    float fov;
};

#endif /* sturg_struct_h */
