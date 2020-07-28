//
//  sturg_loader.cpp
//  sturgLoader
//
//  Created by Dilip Patlolla on 3/24/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#include "sturg_loader.hpp"

// SturgLoader class constructor
SturgLoader::SturgLoader() {
    x_center_coord_ = 553280;   // default values
    y_center_coord_ = 4183397;  // default values
    radius_ = 500;              // default values
    scene_width_ = 1920;        // default values
    scene_height_ = 1080;       // default values
    color_count_ = 0;
}

int SturgLoader::init(SturgInputParams inputData) {
    center_x_ = inputData.center_x;
    center_y_ = inputData.center_y;

    // compute the actual center wrt to tile SIZE
    x_center_coord_ = center_x_ - fmod(center_x_, TILE_SIZE);
    y_center_coord_ = center_y_ - fmod(center_y_, TILE_SIZE);

    model_dir_path_ = inputData.model_dir;
    terrain_dir_path_ = inputData.terrain_dir;

#ifdef VERBOSE
    std::cout.precision(16);
    std::cout << "center_x_: " << center_x_ << ",";
    std::cout << "center_y_: " << center_y_ << std::endl;
    std::cout << "x_center_coord_: " << x_center_coord_ << ",";
    std::cout << "y_center_coord_: " << y_center_coord_ << std::endl;
#endif

    radius_ = inputData.radius;
    scene_width_ = inputData.scene_width;
    scene_height_ = inputData.scene_height;

    // image_width_ and image_height_ are respectively scene_width_/4 and scene_height_/4
    // if no input image_width_ and image_height_ are provided
    image_width_ = (inputData.image_width == 0) ? (scene_width_ / 4) : inputData.image_width;
    image_height_ = (inputData.image_height == 0) ? (scene_height_ / 4) : inputData.image_height;

    fov_ = inputData.fov;
    utm_prefix_ = inputData.utm_prefix;

    // std::cout << "using utm prefix: " << utm_prefix_ << std::endl;
    write_output_ = inputData.write_output;
    color_count_ = 0;
    return 1;
}

std::vector<GLfloat> SturgLoader::getVertices() { return indexed_vertices_; }

std::vector<GLfloat> SturgLoader::getColors() { return indexed_colors_; }

std::vector<GLuint> SturgLoader::getIndices() { return indices_; }

// encode : x,y -> unique combination
double SturgLoader::encode(double param_a, double param_b) {
    return (param_a + param_b) * ((param_a + param_b + 1) / 2) + param_a;
}

// get the optimal buffer tile count
int SturgLoader::getTileCount() {
    // compute tiles required for the given radius
    no_of_tiles_ = int(radius_ / TILE_SIZE);
    no_of_tiles_ = (no_of_tiles_ == 0) ? 1 : no_of_tiles_;

    // minimum tiles required in x and y direction, is 3 if(no_of_tiles == 0)
    no_of_optimum_tiles_ = std::fmax(int(3), no_of_tiles_ * 2 + 1);
    return 1;
}

// compute the origins of tiles encompassed by the radius
int SturgLoader::getTileIds() {
    // compute required tile count
    getTileCount();

    // moving to the left upper corner of area
    x_origin_ = x_center_coord_ - no_of_tiles_ * TILE_SIZE;
    y_origin_ = y_center_coord_ - no_of_tiles_ * TILE_SIZE;

#ifdef VERBOSE
    std::cout << "no_of_tiles_: " << no_of_tiles_ << std::endl;
    std::cout << "no_of_optimum_tiles_: " << no_of_optimum_tiles_ << std::endl;
    std::cout << "x_origin: " << x_origin_ << ",";
    std::cout << "y_origin: " << y_origin_ << std::endl;
#endif

    // compute origins of all the required tiles
    for (int64_t i_iter = 0; i_iter < no_of_optimum_tiles_; i_iter++) {
        for (int64_t j_iter = 0; j_iter < no_of_optimum_tiles_; j_iter++) {
            tile_ids_.push_back(
                this->encode(x_origin_ + i_iter * TILE_SIZE, y_origin_ + j_iter * TILE_SIZE));
        }
    }

    return 1;
}

int SturgLoader::displayTileInfo() {
    if (tile_ids_.empty()) {
        std::cout << "Tile data is empty" << std::endl;
        return 0;
    }

    std::cout << "total tiles:\t" << tile_ids_.size() << std::endl;

    // output tile data
    for (const uint64_t &iter : tile_ids_) {
        std::cout << iter << std::endl;
    }

    return 1;
}

int SturgLoader::displayBinaryFileMeta(const SturGTileMetaData &bin_file_model_info) {
    std::cout << "meta_length :" << bin_file_model_info.meta_length << std::endl;
    std::cout << "models_count :" << bin_file_model_info.models_count << std::endl;
    std::cout << "tile_center_x :" << bin_file_model_info.tile_center_x << std::endl;
    std::cout << "tile_center_y :" << bin_file_model_info.tile_center_y << std::endl;

    return 1;
}

int SturgLoader::getDataFromTiles() {
    // currently reading from local path:
    // TO DO : Read data from input arg..
    if (tile_ids_.empty()) {
        std::cout << "Tile data not available" << std::endl;
        return 0;
    }

    // declare an iterator to a std::vector of uint64_t
    std::vector<uint64_t>::iterator iter;

    // read sturg bin tiles
    for (iter = tile_ids_.begin(); iter < tile_ids_.end(); iter++) {
//        readSturgBinFile(model_dir_path_ + utm_prefix_ + std::to_string(*iter), 0);
        readSturgBinFile(model_dir_path_ + "10N" + std::to_string(*iter), 0);
#ifdef CNN
        readSturgBinFile(terrain_dir_path_ + utm_prefix_ + std::to_string(*iter), 1);
#endif
    }
    return 1;
}

/*int downloadTile(uint32_t tile_id_) {
        // TO DO : download data from amazon s3 bucket
        return 1;
}*/

int SturgLoader::readSturgBinFile(const std::string file_name, unsigned int is_terrain) {
    uint32_t is_uint16;
    SturGVertex vertex;
    SturGFace face;
    SturGBuildingData temp_building;
    std::vector<SturGBuildingData> buildings;
    SturGTileMetaData bin_file_meta;

    std::ifstream tile_file(file_name, std::ios::in | std::ios::binary);
    if (tile_file.is_open()) {
        // get meta data
        tile_file.read(reinterpret_cast<char *>(&bin_file_meta.version), sizeof(uint32_t));
        tile_file.read(reinterpret_cast<char *>(&bin_file_meta.meta_length), sizeof(uint32_t));
        tile_file.read(reinterpret_cast<char *>(&bin_file_meta.models_count), sizeof(uint32_t));
        tile_file.read(reinterpret_cast<char *>(&bin_file_meta.tile_center_x), sizeof(uint32_t));
        tile_file.read(reinterpret_cast<char *>(&bin_file_meta.tile_center_y), sizeof(uint32_t));
        tile_file.read(reinterpret_cast<char *>(&bin_file_meta.tile_center_z), sizeof(uint32_t));

#ifdef VERBOSE2
        // display meta data for convinience
        displayBinaryFileMeta(bin_file_meta);
#endif
        // get models info
        for (uint32_t i = 0; i < bin_file_meta.models_count; i++) {
            tile_file.read(reinterpret_cast<char *>(&temp_building.id), sizeof(uint64_t));
            tile_file.read(reinterpret_cast<char *>(&temp_building.vertices_byte_length),
                           sizeof(uint32_t));
            tile_file.read(reinterpret_cast<char *>(&temp_building.faces_byte_length),
                           sizeof(uint32_t));
            tile_file.read(reinterpret_cast<char *>(&is_uint16), sizeof(uint32_t));
            temp_building.is_uint_16 = bool(is_uint16);

            temp_building.vertices_data_type_size =
                bool(is_uint16) ? sizeof(int16_t) : sizeof(int32_t);

            temp_building.is_terrain = is_terrain;

            buildings.push_back(temp_building);
        }

        // declare an iterator to a std::vector
        std::vector<SturGBuildingData>::iterator iter;

        for (iter = buildings.begin(); iter != buildings.end(); iter++) {
            // TO DO: better refactoring for 32 bit support
            iter->count_vertices =
                uint32_t(iter->vertices_byte_length / iter->vertices_data_type_size / 3);

            for (uint32_t i = 0; i < iter->count_vertices; i++) {
                if (iter->vertices_data_type_size == 2) {
                    int16_t temp_a, temp_b, temp_c;

                    tile_file.read(reinterpret_cast<char *>(&temp_a),
                                   iter->vertices_data_type_size);
                    tile_file.read(reinterpret_cast<char *>(&temp_b),
                                   iter->vertices_data_type_size);
                    tile_file.read(reinterpret_cast<char *>(&temp_c),
                                   iter->vertices_data_type_size);

                    vertex.vertex_param[0] =
                        float(temp_a) / 100.0 + bin_file_meta.tile_center_x - center_x_;
                    vertex.vertex_param[1] =
                        float(temp_b) / 100.0 + bin_file_meta.tile_center_y - center_y_;
                    vertex.vertex_param[2] =
                        float(temp_c) / 100.0 + bin_file_meta.tile_center_z;  // - center_z_;
                } else {
                    int32_t temp_a, temp_b, temp_c;

                    tile_file.read(reinterpret_cast<char *>(&temp_a),
                                   iter->vertices_data_type_size);
                    tile_file.read(reinterpret_cast<char *>(&temp_b),
                                   iter->vertices_data_type_size);
                    tile_file.read(reinterpret_cast<char *>(&temp_c),
                                   iter->vertices_data_type_size);

                    vertex.vertex_param[0] =
                        float(temp_a) / 100.0 + bin_file_meta.tile_center_x - center_x_;
                    vertex.vertex_param[1] =
                        float(temp_b) / 100.0 + bin_file_meta.tile_center_y - center_y_;
                    vertex.vertex_param[2] =
                        float(temp_c) / 100.0 + bin_file_meta.tile_center_z;  // - center_z_;
                }

                iter->vertices.push_back(vertex);
            }

            iter->count_faces = uint32_t(iter->faces_byte_length / sizeof(int16_t) / 3);
            for (uint32_t i = 0; i < iter->count_faces; i++) {
                tile_file.read(reinterpret_cast<char *>(&(face.face_vertex[0])), sizeof(uint16_t));
                tile_file.read(reinterpret_cast<char *>(&(face.face_vertex[1])), sizeof(uint16_t));
                tile_file.read(reinterpret_cast<char *>(&(face.face_vertex[2])), sizeof(uint16_t));
                iter->faces.push_back(face);
            }
        }

        // concatenate the buildings from the current file to the global buildings_ data
        buildings_.reserve(buildings_.size() + buildings.size());
        std::move(buildings.begin(), buildings.end(), std::inserter(buildings_, buildings_.end()));
        // buildings.clear();
        tile_file.close();
    } else {
        std::cout << " >>> skipping file:\t" << file_name << std::endl;
    }

    return 1;
}


int SturgLoader::displayBuildingsData() {
    for (const SturGBuildingData &building_iter : buildings_) {
        std::cout << "Building ID: " << building_iter.id << "\t";
        std::cout << "No. of vertices: " << building_iter.vertices.size() << "\t";
        std::cout << "No. of faces: " << building_iter.faces.size() << std::endl;
        std::cout << "UINT16: " << building_iter.is_uint_16;
        std::cout << "\tVertices Byte Length: " << building_iter.vertices_byte_length;
        std::cout << "\tFaces Byte Length: " << building_iter.faces_byte_length << std::endl;
#ifdef VERBOSE3
        std::cout << "Vertices : " << std::endl;
        for (const SturGVertex &vertice_iter : building_iter.vertices) {
            std::cout << "[" << vertice_iter.vertex_param[0] << " " << vertice_iter.vertex_param[1]
                      << " " << vertice_iter.vertex_param[2] << "]" << std::endl;
        }

        std::cout << "Faces : " << std::endl;
        for (const SturGFace &face_iter : building_iter.faces) {
            std::cout << "[" << face_iter.face_vertex[0] << " " << face_iter.face_vertex[1] << " "
                      << face_iter.face_vertex[2] << "]" << std::endl;
        }
#endif
    }

    return 1;
}

int SturgLoader::process() {
    processDataforRendering();
    return 1;
}

int SturgLoader::processDataforRendering() {
    int j;

    // reset color count
    color_count_ = 0;

    // std::cout << "\getting Tile Ids and reading data from them\n" << std::endl;
    getTileIds();
    getDataFromTiles();

    // temp variable to store color params
    float rand_color_array[COLOR_PARAM_SIZE] = {1.0f, 1.0f, 1.0f};

    if (buildings_.empty()) {
        std::cout << "Building data is empty" << std::endl;
        return 0;
    }

#ifdef VERBOSE
    displayBuildingsData();
#endif

    for (SturGBuildingData &building_iter : buildings_) {
        std::vector<SturGVertex>::iterator vertice_iter = building_iter.vertices.begin();
        getRandomColor(rand_color_array, building_iter.id, building_iter.is_terrain);

        // for each face get vertices and corresponding colors
        for (const SturGFace &face_iter : building_iter.faces) {
            for (int i = 0; i < FACE_VERTICES_SIZE; i++) {
                vertice_iter = building_iter.vertices.begin() + face_iter.face_vertex[i];
                if (!building_iter.is_terrain) {
                    vertices_.push_back(*vertice_iter);

                    for (j = 0; j < VERTEX_VERTICES_SIZE; j++) {
                        colors_.push_back(rand_color_array[j]);
                        // To DO: needs fix if VERTEX_VERTICES_SIZE!=COLOR_PARAM_SIZE
                    }
                    if (COLOR_PARAM_SIZE == 4) colors_.push_back(0.0);
                }

                else {
                    ter_vertices_.push_back(*vertice_iter);

                    for (j = 0; j < VERTEX_VERTICES_SIZE; j++) {
                        ter_colors_.push_back(rand_color_array[j]);
                        // To DO: needs fix if VERTEX_VERTICES_SIZE!=COLOR_PARAM_SIZE
                    }
                    if (VERTEX_VERTICES_SIZE == 4) ter_colors_.push_back(0.0);
                }
            }
        }
    }

    GLuint max = this->computeIndices(vertices_, colors_, indices_, vertex_to_index_map_,
                                      indexed_vertices_, indexed_colors_, 0, 0);

#ifdef CNN
    this->computeIndices(ter_vertices_, ter_colors_, ter_indices_, ter_vertex_to_index_map_,
                         ter_indexed_vertices_, ter_indexed_colors_, max + 1, 1);

    // copy the terrain data into global data used for rendering
    indexed_vertices_.insert(indexed_vertices_.end(),
                             std::make_move_iterator(ter_indexed_vertices_.begin()),
                             std::make_move_iterator(ter_indexed_vertices_.end()));

    indexed_colors_.insert(indexed_colors_.end(),
                           std::make_move_iterator(ter_indexed_colors_.begin()),
                           std::make_move_iterator(ter_indexed_colors_.end()));

    indices_.insert(indices_.end(), std::make_move_iterator(ter_indices_.begin()),
                    std::make_move_iterator(ter_indices_.end()));
#endif

#ifdef VERBOSE
    auto iter = max_element(std::begin(indices_), std::end(indices_));  // c++11
    std::cout << "max afer ind: " << *iter << std::endl;

    std::cout << "vertices size :" << vertices_.size() << std::endl;
    std::cout << "color size :" << colors_.size() << std::endl;
    std::cout << "indices size :" << indices_.size() << std::endl;
    std::cout << "index vertices size :" << indexed_vertices_.size() << std::endl;
    std::cout << "index color size :" << indexed_colors_.size() << std::endl;

    std::cout << "terrain vertices size :" << ter_vertices_.size() << std::endl;
    std::cout << "terrain color size :" << ter_colors_.size() << std::endl;
    std::cout << "terrain indices size :" << ter_indices_.size() << std::endl;
    std::cout << "terrain index vertices size :" << ter_indexed_vertices_.size() << std::endl;
    std::cout << "terrain index color size :" << ter_indexed_colors_.size() << std::endl;
#endif

    return 1;
}

int SturgLoader::displayColors() {
    std::ofstream outputFile;
    outputFile.open("colors.csv");
    outputFile.precision(8);

    for (auto pp_iter = colors_.begin(); pp_iter != indexed_colors_.end(); pp_iter++) {
        outputFile << *pp_iter << ",";
    }
    outputFile.close();
    return 1;
}

int SturgLoader::displayVertices() {
    std::ofstream outputFile;
    outputFile.open("vertices.csv");
    outputFile.precision(8);

    std::cout << "index vertices size :" << indexed_vertices_.size() << std::endl;
    for (auto pp_iter = indexed_vertices_.begin(); pp_iter != indexed_vertices_.end(); pp_iter++) {
        outputFile << *pp_iter << ",";
    }
    outputFile.close();

    return 1;
}

int SturgLoader::displayIndices() {
    std::ofstream outputFile;
    outputFile.open("indices.csv");
    outputFile.precision(8);

    std::cout << "indices size :" << indices_.size() << std::endl;
    for (auto pp_iter = indices_.begin(); pp_iter != indices_.end(); pp_iter++) {
        outputFile << *pp_iter << std::endl;
    }
    outputFile.close();

    return 1;
}

int SturgLoader::getRandomColor(float rand_color_array[], uint64_t seed, unsigned int is_terrain) {
    // generate random for buildings from model files
    // the result is unchanged if its for cnn
    if (is_terrain) {
        rand_color_array[0] = 0.5f;
        rand_color_array[1] = 0.5f;
        rand_color_array[2] = 0.5f;
    } else {
        // std::cout << colorsTable[color_count_ % NUM_OF_COLORS][1] << std::endl;
        // input unique seed
        // srand(static_cast<int>(seed * time(0)));
        // generate random numbers between 0 to 255;
        // making sure  R band is never zero
        // convert them to OpenGL colors float format
        rand_color_array[0] =
            colorsTable[color_count_ % NUM_OF_COLORS][0];  //(((rand() + 1) % 255) / 255.0f);
        rand_color_array[1] =
            colorsTable[color_count_ % NUM_OF_COLORS][1];  //(((rand() + 2) % 255) / 255.0f);
        rand_color_array[2] =
            colorsTable[color_count_ % NUM_OF_COLORS][2];  //(((rand() + 3) % 255) / 255.0f);

        // Making sure its a non zero value and non terrain color value
        // green
        // while(rand_color_array[1] == 0.5f || rand_color_array[1] == 0.0f){
        //    rand_color_array[1] = (((rand() + 2) % 255) / 255.0f);
        //}
        // blue
        // while(rand_color_array[2] == 0.5f || rand_color_array[2] == 0.0f){
        //    rand_color_array[2] = (((rand() + 3) % 255) / 255.0f);
        //}

        // for memory access we are using just color value in r band position
        // rand_color_array[0] = (rand_color_array[0]*0.21 + rand_color_array[1]*0.72 +
        // rand_color_array[2]*0.07);

        // while(rand_color_array[0] == 0.5f || rand_color_array[0] == 0.0f){
        //    rand_color_array[0] = (((rand() + 1) % 255) / 255.0f);
        //    rand_color_array[0] = (rand_color_array[0]*0.21 + rand_color_array[1]*0.72 +
        //    rand_color_array[2]*0.07);
        //}
    }

    color_count_++;
    return 1;
}

bool SturgLoader::checkForSimilarVertex(SturGVertex &vertex_in, GLuint &result,
                                        unsigned int is_terrain) {
    if (!is_terrain) {
        auto vertex_map_iter = vertex_to_index_map_.find(vertex_in);
        auto end = vertex_to_index_map_.end();

        if (vertex_map_iter == end) {
            return false;
        } else {
            result = vertex_map_iter->second;
            return true;
        }
    } else {
        auto vertex_map_iter = ter_vertex_to_index_map_.find(vertex_in);
        auto end = ter_vertex_to_index_map_.end();
        if (vertex_map_iter == end) {
            return false;
        } else {
            result = vertex_map_iter->second;
            return true;
        }
    }
}

// TO DO : ccompute these at the time of loading in  processDataforRendering();
GLuint SturgLoader::computeIndices(std::vector<SturGVertex> vertices, std::vector<GLfloat> colors,
                                   std::vector<GLuint> &indices,
                                   std::map<SturGVertex, GLuint> &vertex_to_index_map,
                                   std::vector<GLfloat> &indexed_vertices,
                                   std::vector<GLfloat> &indexed_colors, GLuint offset,
                                   unsigned int is_terrain) {
    // std::cout << "vertices size: " << vertices_.size() << std::endl;
    unsigned int max = 0;
    // for (auto pp_iter = vertices_.begin(); pp_iter != vertices_.end(); pp_iter++,i++) {
    for (GLuint i = 0; i < vertices.size(); i++) {
        // check if the vertex already exists
        GLuint index;
        bool found =
            checkForSimilarVertex(vertices[i], index, is_terrain);  //, vertex_to_index_map_);

        if (found) {  // vertex is already in the VBO, so update index only
            indices.push_back(index + offset);
        } else {
            // update the indexed vertices
            indexed_vertices.push_back(vertices[i].vertex_param[0]);
            indexed_vertices.push_back(vertices[i].vertex_param[1]);
            indexed_vertices.push_back(vertices[i].vertex_param[2]);

            // update the indexed colors
            for (unsigned int temp = 0; temp < COLOR_PARAM_SIZE; temp++) {
                indexed_colors.push_back(colors[i * COLOR_PARAM_SIZE + temp]);
            }

            // compute the new index and uodate the indices used for rendering
            GLuint new_index = (GLuint)indexed_vertices.size() / VERTEX_VERTICES_SIZE - 1;
            indices.push_back(new_index + offset);

            // update max when a new index is computed
            max = (max < new_index) ? new_index : max;
            vertex_to_index_map[vertices[i]] = new_index;
        }
    }
    return max;
}

// SturgLoader class destructor with input params
SturgLoader::~SturgLoader() {
    // at expense of speed
    std::vector<SturGBuildingData>().swap(buildings_);
    std::vector<uint64_t>().swap(tile_ids_);
    std::vector<SturGVertex>().swap(vertices_);
    std::vector<GLuint>().swap(indices_);
    std::vector<GLfloat>().swap(colors_);
    std::vector<GLfloat>().swap(indexed_vertices_);
    std::vector<GLfloat>().swap(indexed_colors_);
    std::vector<SturGVertex>().swap(ter_vertices_);
    std::vector<GLuint>().swap(ter_indices_);
    std::vector<GLfloat>().swap(ter_colors_);
    std::vector<GLfloat>().swap(ter_indexed_vertices_);
    std::vector<GLfloat>().swap(ter_indexed_colors_);
}
