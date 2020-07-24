//
//  sturg_loader.hpp
//  sturgLoader
//
//  Created by Dilip Patlolla on 3/24/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#ifndef STURG_LOADER_HPP
#define STURG_LOADER_HPP

#include <stdio.h>
#include <algorithm>  // std::max
#include <cmath>      // std::ceil
#include <ctime>      // time
#include <fstream>    // infile
#include <iostream>   // std::std::cout
#include <iterator>   // std:: distance
#include <list>       // std::list<>
#include <map>        // std::map<>
#include <sstream>    // ss
#include <vector>     // std::vector<>

#include "GL/glew.h"  // GLfloat

// sturg specific
#include "sturg_colors.h"
#include "sturg_helper_func.hpp"
#include "sturg_struct.h"
// utm
//#include "utm/utm_from_lat_long.hpp"

#define SCENE_WIDTH 1920
#define SCENE_HEIGHT 1080

class SturgLoader {
   public:
    // constructors
    SturgLoader();

    int init(SturgInputParams inputData);
    int process();
    std::vector<GLfloat> getVertices();
    std::vector<GLfloat> getColors();
    std::vector<GLuint> getIndices();

    // display
    int displayTileInfo();
    int displayBuildingsData();
    int displayVertices();
    int displayColors();
    int displayIndices();
    int displayIndexVertices();

    // destructor
    ~SturgLoader();

   private:
    std::string model_dir_path_;
    std::string terrain_dir_path_;
    uint64_t x_center_coord_, y_center_coord_;
    double center_x_, center_y_;
    uint64_t x_origin_, y_origin_;
    float pos_z_, radius_, fov_;
    int no_of_tiles_, no_of_optimum_tiles_;
    int scene_width_, scene_height_;
    int image_width_, image_height_;
    bool write_output_;
    int color_count_;

    std::string utm_prefix_;

    std::vector<SturGBuildingData> buildings_;
    std::vector<uint64_t> tile_ids_;

    std::vector<GLuint> indices_;
    std::vector<GLfloat> colors_;
    std::vector<SturGVertex> vertices_;
    std::map<SturGVertex, GLuint> vertex_to_index_map_;
    std::vector<GLfloat> indexed_vertices_;
    std::vector<GLfloat> indexed_colors_;

    std::vector<GLuint> ter_indices_;
    std::vector<GLfloat> ter_colors_;
    std::vector<SturGVertex> ter_vertices_;
    std::map<SturGVertex, GLuint> ter_vertex_to_index_map_;
    std::vector<GLfloat> ter_indexed_vertices_;
    std::vector<GLfloat> ter_indexed_colors_;

    double encode(double param_a, double param_b);
    int displayBinaryFileMeta(const SturGTileMetaData& bin_file_model_info);
    // int downloadTile(uint64_t tile_id_);
    int getTileCount();
    int getTileIds();
    int getDataFromTiles();
    int readSturgBinFile(const std::string file_name, unsigned int is_terrain);
    int processDataforRendering();
    int getRandomColor(float rand_color[], uint64_t model_id, unsigned int is_terrain);
    bool checkForSimilarVertex(SturGVertex& vertex_in, GLuint& result, unsigned int is_terrain);
    // const std::map<SturGVertex,GLuint> vertex_map);
    GLuint computeIndices(std::vector<SturGVertex> vertices_, std::vector<GLfloat> colors_,
                          std::vector<GLuint>& indices,
                          std::map<SturGVertex, GLuint>& vertex_to_index_map,
                          std::vector<GLfloat>& indexed_vertices,
                          std::vector<GLfloat>& indexed_colors, GLuint offset,
                          unsigned int is_terrain);
};

#endif /* STURG_LOADER_HPP */
