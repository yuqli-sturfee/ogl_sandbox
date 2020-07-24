//
//  rc_loader.hpp
//  rc_loader
//
//  Created by Dilip Patlolla on 09/21/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#ifndef RC_LOADER_HPP
#define RC_LOADER_HPP

#include <ctime>    // time
#include <fstream>  // infile
#include <iomanip>  // precision
#include <map>      // std::map<>
#include <map>
#include <sstream>  // ss

#include <cuda.h>  // cuda

#include "GL/glew.h"  // GLfloat

// sturg specific
#include "rc_struct.h"
#include "sturg_struct.h"

#include "point_in_triangle.hpp"

#define SPATIAL_FILTER

class RcLoader {
   public:
    // constructors
    RcLoader();

    int initSf(SturgInputParams input_data);
    std::vector<SturgCameraParameters> processSf(std::vector<SturgCameraParameters> all_cam_params);

    std::vector<GLfloat> getColors();
    std::vector<GLfloat> getVertices();
    std::vector<SturGVertex> getRawVertices();

    std::vector<float3> getTriangles();
    MbrRectangles getRectangles();
    MbrFaces getFaces();
    std::map<double, MbrFaces> getMappedFaces();
    MbrLsForSF getMbrsAsVector();

    // display
    int displayTileInfo();
    int displayBuildingsData();
    int displayVertices();
    int displayMaxBindingRectangles();
    unsigned int getTrianglesCount();

    // destructor
    ~RcLoader();

   private:
    std::vector<SturgCameraParameters> filtered_cam_params_;
    uint64_t x_center_coord_, y_center_coord_;
    double center_x_, center_y_;
    uint64_t x_origin_, y_origin_;
    float pos_z_, radius_, fov_;
    int no_of_tiles_, no_of_optimum_tiles_;
    int scene_width_, scene_height_;
    int image_width_, image_height_;
    bool write_output_;
    unsigned int num_triangles_;
    double camera_height_;
    std::vector<SturGBuildingData> buildings_;
    std::vector<uint64_t> tile_ids_;

    std::vector<GLuint> indices_;
    std::vector<GLfloat> colors_;

    std::vector<SturGVertex> vertices_;
    std::vector<GLfloat> raycast_vertices_;

    std::vector<GLfloat> indexed_vertices_;
    std::vector<GLfloat> indexed_colors_;

    std::vector<GLfloat> ter_colors_;
    std::vector<SturGVertex> ter_vertices_;

    std::vector<float3> triangles_;
    MbrRectangles pid_MB_Rectangles_;
    MbrLsForSF max_binding_rectangles_;

    // empty map container
    std::map<double, MbrFaces> pid_and_faces_;
    std::map<std::pair<double, double>, std::vector<size_t>> x_y_face_intersect_;

    double encode(double param_a, double param_b);
    int displayBinaryFileMeta(const SturGTileMetaData& bin_file_model_info);
    int getTileCount();
    int getTileIds();
    int getDataFromTiles();
    int readSturgBinFile(const std::string file_name, unsigned int is_terrain);
    int readData();

    int getRandomColor(float rand_color[], uint64_t model_id, unsigned int is_terrain);

    int processDataforRendering();
    MbrRectangle getMbrRectangle(const std::vector<SturGVertex>);
    int unRavelFaces(SturGBuildingData data);
};

#endif /* RC_LOADER_HPP */
