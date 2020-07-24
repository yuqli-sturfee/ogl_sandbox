//
//  rc_loader.cpp
//  RcLoader
//
//  Created by Dilip Patlolla on 09/21/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#include "rc_loader.hpp"

// RcLoader class constructor
RcLoader::RcLoader() {
    x_center_coord_ = 553280;   // default values
    y_center_coord_ = 4183397;  // default values
    radius_ = 500;              // default values
    scene_width_ = 1920;        // default values
    scene_height_ = 1080;       // default values
    num_triangles_ = 0;
    camera_height_ = 0;
}

int RcLoader::initSf(SturgInputParams inputData) {
    center_x_ = inputData.center_x;
    center_y_ = inputData.center_y;

    // compute the actual center wrt to tile SIZE
    x_center_coord_ = center_x_ - fmod(center_x_, TILE_SIZE);
    y_center_coord_ = center_y_ - fmod(center_y_, TILE_SIZE);

#ifdef VERBOSE_LOADER
    std::cout.precision(16);
    std::cout << "center_x_: " << center_x_ << ",";
    std::cout << "center_y_: " << center_y_ << std::endl;
    std::cout << "x_center_coord_: " << x_center_coord_ << ",";
    std::cout << "y_center_coord_: " << y_center_coord_ << std::endl;
#endif

    radius_ = inputData.radius;
    scene_width_ = inputData.scene_width;
    scene_height_ = inputData.scene_height;
    camera_height_ = inputData.cam_height;
    // image_width_ and image_height_ are respectively scene_width_/4 and scene_height_/4
    // if no input image_width_ and image_height_ are provided
    // image_width_ = (inputData.image_width == 0) ? (scene_width_/4) : inputData.image_width;
    // image_height_ = (inputData.image_height == 0) ? (scene_height_/4) : inputData.image_height;

    fov_ = inputData.fov;
    write_output_ = inputData.write_output;
    num_triangles_ = 0;
    return 1;
}

std::vector<SturGVertex> RcLoader::getRawVertices() { return vertices_; }

MbrLsForSF RcLoader::getMbrsAsVector() {
    max_binding_rectangles_.min_vertex_x.clear();
    max_binding_rectangles_.max_vertex_x.clear();
    max_binding_rectangles_.min_vertex_y.clear();
    max_binding_rectangles_.max_vertex_y.clear();

    // TO DO: check here if the vertices_ has been populated
    for (auto i = 0; i < vertices_.size(); i += 3) {
        max_binding_rectangles_.min_vertex_x.push_back(
            std::min({vertices_[i + 0].vertex_param[0], vertices_[i + 1].vertex_param[0],
                      vertices_[i + 2].vertex_param[0]}));
        max_binding_rectangles_.min_vertex_y.push_back(
            std::min({vertices_[i + 0].vertex_param[1], vertices_[i + 1].vertex_param[1],
                      vertices_[i + 2].vertex_param[1]}));
        max_binding_rectangles_.max_vertex_x.push_back(
            std::max({vertices_[i + 0].vertex_param[0], vertices_[i + 1].vertex_param[0],
                      vertices_[i + 2].vertex_param[0]}));
        max_binding_rectangles_.max_vertex_y.push_back(
            std::max({vertices_[i + 0].vertex_param[1], vertices_[i + 1].vertex_param[1],
                      vertices_[i + 2].vertex_param[1]}));
    }

    return max_binding_rectangles_;
}

std::vector<GLfloat> RcLoader::getVertices() { return raycast_vertices_; }

std::vector<float3> RcLoader::getTriangles() { return triangles_; }

// encode : x,y -> unique combination
double RcLoader::encode(double param_a, double param_b) {
    return (param_a + param_b) * ((param_a + param_b + 1) / 2) + param_a;
}

// get the optimal buffer tile count
int RcLoader::getTileCount() {
    // compute tiles required for the given radius
    no_of_tiles_ = int(radius_ / TILE_SIZE);
    no_of_tiles_ = (no_of_tiles_ == 0) ? 1 : no_of_tiles_;

    // minimum tiles required in x and y direction, is 3 if(no_of_tiles == 0)
    no_of_optimum_tiles_ = std::fmax(int(3), no_of_tiles_ * 2 + 1);
    return 1;
}

// compute the origins of tiles encompassed by the radius
int RcLoader::getTileIds() {
    // compute required tile count
    getTileCount();

    // moving to the left upper corner of area
    x_origin_ = x_center_coord_ - no_of_tiles_ * TILE_SIZE;
    y_origin_ = y_center_coord_ - no_of_tiles_ * TILE_SIZE;

#ifdef VERBOSE_LOADER
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

int RcLoader::displayTileInfo() {
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

int RcLoader::displayBinaryFileMeta(const SturGTileMetaData &bin_file_model_info) {
    std::cout << "meta_length :" << bin_file_model_info.meta_length << std::endl;
    std::cout << "models_count :" << bin_file_model_info.models_count << std::endl;
    std::cout << "tile_center_x :" << bin_file_model_info.tile_center_x << std::endl;
    std::cout << "tile_center_y :" << bin_file_model_info.tile_center_y << std::endl;

    return 1;
}

int RcLoader::getDataFromTiles() {
    // currently reading from local path:
    // TO DO : Read data from input arg..
    if (tile_ids_.empty()) {
        std::cout << "Tile data not available" << std::endl;
        return 0;
    }

#ifdef G2
    std::string terr_dir_path = "/home/ubuntu/geometry_terrain/";

#ifdef SPATIAL_FILTER
    // TO DO: move to s3 implementation
    std::string data_dir_path = "/home/ubuntu/geometry_terrain_holes/";
#else
    // TO DO: move to s3 implementation
    std::string data_dir_path = "/home/ubuntu/geometry_models/";
#endif

#else

    std::string terr_dir_path = "/Users/PDR/Desktop/PROJECTS/sturfee/data/geometry_terrain/";

#ifdef SPATIAL_FILTER
    std::string data_dir_path = "/Users/PDR/Desktop/PROJECTS/sturfee/data/geometry_terrain_holes/";
#else
    std::string data_dir_path = "/Users/PDR/Desktop/PROJECTS/sturfee/data/geometry_models/";
#endif

#endif

    // declare an iterator to a std::vector of uint64_t
    std::vector<uint64_t>::iterator iter;

    // read sturg bin tiles
    for (iter = tile_ids_.begin(); iter < tile_ids_.end(); iter++) {
        readSturgBinFile(data_dir_path + std::to_string(*iter), 0);
#ifdef CNN
        readSturgBinFile(terr_dir_path + std::to_string(*iter), 1);
#endif
    }
    return 1;
}

/*int downloadTile(uint32_t tile_id_) {
        // TO DO : download data from amazon s3 bucket
        return 1;
}*/

int RcLoader::readSturgBinFile(const std::string file_name, unsigned int is_terrain) {
    int16_t temp_a, temp_b, temp_c;
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
#ifdef VERBOSE_LOADER
        // display meta data for convinience
        displayBinaryFileMeta(bin_file_meta);
#endif
        // get models info
        for (uint32_t i = 0; i < bin_file_meta.models_count; i++) {
            tile_file.read(reinterpret_cast<char *>(&temp_building.id), sizeof(double));

            tile_file.read(reinterpret_cast<char *>(&temp_building.vertices_byte_length),
                           sizeof(uint32_t));
            tile_file.read(reinterpret_cast<char *>(&temp_building.faces_byte_length),
                           sizeof(uint32_t));
            tile_file.read(reinterpret_cast<char *>(&is_uint16), sizeof(uint32_t));
            temp_building.is_uint_16 = bool(is_uint16);
            temp_building.is_terrain = is_terrain;
            buildings.push_back(temp_building);
        }

        // declare an iterator to a std::vector
        std::vector<SturGBuildingData>::iterator iter;

        for (iter = buildings.begin(); iter != buildings.end(); iter++) {
            // TO DO: better refactoring for 32 bit support
            iter->count_vertices = uint32_t(iter->vertices_byte_length / sizeof(u_int16_t) / 3);

            for (uint32_t i = 0; i < iter->count_vertices; i++) {
                tile_file.read(reinterpret_cast<char *>(&temp_a), sizeof(int16_t));
                tile_file.read(reinterpret_cast<char *>(&temp_b), sizeof(int16_t));
                tile_file.read(reinterpret_cast<char *>(&temp_c), sizeof(int16_t));
                vertex.vertex_param[0] =
                    float(temp_a) / 100.0 + bin_file_meta.tile_center_x - center_x_;
                vertex.vertex_param[1] =
                    float(temp_b) / 100.0 + bin_file_meta.tile_center_y - center_y_;
                vertex.vertex_param[2] = float(temp_c) / 100.0;
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
        std::cout << "skipping file:\t" << file_name << std::endl;
    }

    return 1;
}

int RcLoader::displayBuildingsData() {
    for (const SturGBuildingData &building_iter : buildings_) {
        std::cout << "Building ID: " << building_iter.id << "\t";
        std::cout << "No. of vertices: " << building_iter.vertices.size() << "\t";
        std::cout << "No. of faces: " << building_iter.faces.size() << std::endl;
        std::cout << "Is terrain: " << building_iter.is_terrain << std::endl;
        std::cout << "UINT16: " << building_iter.is_uint_16;
        std::cout << "\tVertices Byte Length: " << building_iter.vertices_byte_length;
        std::cout << "\tFaces Byte Length: " << building_iter.faces_byte_length << std::endl;
#ifdef VERBOSE_LOADER2
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
std::vector<SturgCameraParameters> RcLoader::processSf(
    std::vector<SturgCameraParameters> all_cam_params) {
    processDataforRendering();
    readData();
    filtered_cam_params_.clear();

    SturgCameraParameters temp_cam_param;
    // MbrRectangles pid_MB_Rectangles = this->getRectangles();
    unsigned int max_triangles_count = this->getTrianglesCount();
    std::vector<SturGVertex> vertices = this->getRawVertices();
    // MbrLsForSF max_binding_rectangles = this->getMbrsAsVector();

    // double x = 0;
    // double y = 0;
    // float height = 0;

    // std::vector<size_t> x_min_vertices;  // (max_binding_rectangles.min_vertex_x.size());
    // std::vector<size_t> y_min_vertices;  // (max_binding_rectangles.min_vertex_y.size());
    // std::vector<size_t> x_max_vertices;  // (max_binding_rectangles.max_vertex_x.size());
    // std::vector<size_t> y_max_vertices;  // (max_binding_rectangles.max_vertex_y.size());
    // std::vector<size_t> x_intersection, y_intersection;
    // std::vector<size_t> face_intersection;

    // // std::cout << "size of all cam params" << all_cam_params.size() << std::endl;
    // unsigned int count = 0;
    // for (auto iter = all_cam_params.begin(); iter < all_cam_params.end(); iter++) {
    //     x = iter->cam_x - center_x_;
    //     y = iter->cam_y - center_y_;

    //     // check if the building id already exists in the map
    //     auto x_y_iter = x_y_face_intersect_.find(std::make_pair(x, y));

    //     // not found
    //     if (x_y_iter == x_y_face_intersect_.end()) {
    //         // TO DO : from here on, should be in loop
    //         for (std::size_t index = 0; index < max_binding_rectangles.min_vertex_x.size();
    //              ++index) {
    //             if (max_binding_rectangles.min_vertex_x[index] <= x) {
    //                 x_min_vertices.push_back(index);
    //             }
    //             if (max_binding_rectangles.max_vertex_x[index] >= x) {
    //                 x_max_vertices.push_back(index);
    //             }
    //             if (max_binding_rectangles.min_vertex_y[index] <= y) {
    //                 y_min_vertices.push_back(index);
    //             }
    //             if (max_binding_rectangles.max_vertex_y[index] >= y) {
    //                 y_max_vertices.push_back(index);
    //             }
    //         }

    //         std::set_intersection(x_min_vertices.begin(), x_min_vertices.end(),
    //                               x_max_vertices.begin(), x_max_vertices.end(),
    //                               std::back_inserter(x_intersection));
    //         std::set_intersection(y_min_vertices.begin(), y_min_vertices.end(),
    //                               y_max_vertices.begin(), y_max_vertices.end(),
    //                               std::back_inserter(y_intersection));
    //         std::set_intersection(x_intersection.begin(), x_intersection.end(),
    //                               y_intersection.begin(), y_intersection.end(),
    //                               std::back_inserter(face_intersection));

    //         x_y_face_intersect_.insert(std::make_pair(std::make_pair(x, y), face_intersection));

    //         // clear temp min/max vectors
    //         x_min_vertices.clear();
    //         y_min_vertices.clear();
    //         x_max_vertices.clear();
    //         y_max_vertices.clear();
    //         x_intersection.clear();
    //         y_intersection.clear();
    //     }

    //     // found
    //     else {
    //         face_intersection = x_y_iter->second;
    //     }

    //     // std::cout << std::setprecision(5) << iter->cam_x << "\t" << iter->cam_y << std::endl;
    //     // std::cout << "face_intersection: " << face_intersection.size() << std::endl;
    //     count++;

    //     for (auto i : face_intersection) {
    //         height = point_in_triangle(i, x, y, vertices);
    //         if (height != NO_HIT) {
    //             // std::cout << x << "\t" << y << "\t" << "height: " << height<< " ,cam h: " <<
    //             // camera_height_ <<std::endl;
    //             temp_cam_param = *iter;
    //             temp_cam_param.cam_z = height + camera_height_;
    //             filtered_cam_params_.push_back(temp_cam_param);
    //             break;
    //         }
    //     }

    //     face_intersection.clear();
    // }

    // end the loop here
    return filtered_cam_params_;
}

int RcLoader::readData() {
    MbrRectangle temp_rectangle;

    for (SturGBuildingData &building_iter : buildings_) {
#ifdef VERBOSE_LOADER
        std::cout << building_iter.id << "," << building_iter.vertices.size() << ","
                  << building_iter.faces.size() << std::endl;
#endif
        temp_rectangle = getMbrRectangle(building_iter.vertices);
        pid_MB_Rectangles_.id.push_back(building_iter.id);
        pid_MB_Rectangles_.min_vertices.push_back(temp_rectangle.min_vertices);
        pid_MB_Rectangles_.max_vertices.push_back(temp_rectangle.max_vertices);

        unRavelFaces(building_iter);
    }
#ifdef VERBOSE_LOADER
    std::cout << "building  size: " << buildings_.size() << std::endl;
    std::cout << "mbr size: " << pid_MB_Rectangles_.id.size() << std::endl;
#endif
    return 1;
}

int RcLoader::unRavelFaces(SturGBuildingData data) {
#ifdef VERBOSE_LOADER
    std::cout << data.id << "\t";
    std::cout << "vertices size: " << data.vertices.size() << "\t";
    std::cout << "faces size: " << data.faces.size() << std::endl;
#endif
    float3 temp;
    MbrFaces pid_faces;

    for (auto it = data.faces.begin(); it != data.faces.end(); ++it) {
        // std::cout << it->face_vertex[0] << ",";
        // std::cout << it->face_vertex[1] << ",";
        // std::cout << it->face_vertex[2] << ",";
        // auto index = std::distance(data.faces.begin(), it)

        auto v1 = it->face_vertex[0];
        auto v2 = it->face_vertex[1];
        auto v3 = it->face_vertex[2];

        pid_faces.id.push_back(data.id);

        temp = make_float3(data.vertices[v1].vertex_param[0], data.vertices[v1].vertex_param[1],
                           data.vertices[v1].vertex_param[2]);

        pid_faces.x_face.push_back(temp);

        temp = make_float3(data.vertices[v2].vertex_param[0], data.vertices[v2].vertex_param[1],
                           data.vertices[v2].vertex_param[2]);

        pid_faces.y_face.push_back(temp);

        temp = make_float3(data.vertices[v3].vertex_param[0], data.vertices[v3].vertex_param[1],
                           data.vertices[v3].vertex_param[2]);

        pid_faces.z_face.push_back(temp);

        temp.x = std::min({data.vertices[v1].vertex_param[0], data.vertices[v2].vertex_param[0],
                           data.vertices[v3].vertex_param[0]});
        temp.y = std::min({data.vertices[v1].vertex_param[1], data.vertices[v2].vertex_param[1],
                           data.vertices[v3].vertex_param[1]});
        temp.z = std::min({data.vertices[v1].vertex_param[2], data.vertices[v2].vertex_param[2],
                           data.vertices[v3].vertex_param[2]});

        pid_faces.mbrFace_sub.push_back(temp);
    }

    // check if the building id already exists in the map
    auto map_iter = pid_and_faces_.find(data.id);

    // if exists add the new triangles to the existing ones.
    if (map_iter != pid_and_faces_.end()) {
        map_iter->second.id.insert(map_iter->second.id.end(), pid_faces.id.begin(),
                                   pid_faces.id.end());
        map_iter->second.x_face.insert(map_iter->second.x_face.end(), pid_faces.x_face.begin(),
                                       pid_faces.x_face.end());
        map_iter->second.y_face.insert(map_iter->second.y_face.end(), pid_faces.y_face.begin(),
                                       pid_faces.y_face.end());
        map_iter->second.z_face.insert(map_iter->second.z_face.end(), pid_faces.z_face.begin(),
                                       pid_faces.z_face.end());

    }
    // add new entry
    else {
        pid_and_faces_.insert(std::make_pair(data.id, pid_faces));
    }

    num_triangles_ += pid_faces.x_face.size();

    return 1;
}

MbrRectangles RcLoader::getRectangles() { return pid_MB_Rectangles_; }

std::map<double, MbrFaces> RcLoader::getMappedFaces() { return pid_and_faces_; }

unsigned int RcLoader::getTrianglesCount() { return num_triangles_; }
MbrRectangle RcLoader::getMbrRectangle(const std::vector<SturGVertex> vertices) {
    MbrRectangle temp_rectangle;

    auto minmax_vertices = std::minmax_element(vertices.begin(), vertices.end(),
                                               [](SturGVertex const &lhs, SturGVertex const &rhs) {
                                                   return lhs.vertex_param[0] < rhs.vertex_param[0];
                                               });
    temp_rectangle.min_vertices.x = minmax_vertices.first->vertex_param[0];
    temp_rectangle.max_vertices.x = minmax_vertices.second->vertex_param[0];

    // std::cout <<  "min: " << minmax_vertices.first->vertex_param[0] << " max: " <<
    // minmax_vertices.second->vertex_param[0] << std::endl;

    minmax_vertices = std::minmax_element(vertices.begin(), vertices.end(),
                                          [](SturGVertex const &lhs, SturGVertex const &rhs) {
                                              return lhs.vertex_param[1] < rhs.vertex_param[1];
                                          });
    temp_rectangle.min_vertices.y = minmax_vertices.first->vertex_param[1];
    temp_rectangle.max_vertices.y = minmax_vertices.second->vertex_param[1];

    // std::cout <<  "min: " << minmax_vertices.first->vertex_param[1] << " max: " <<
    // minmax_vertices.second->vertex_param[1] << std::endl;

    minmax_vertices = std::minmax_element(vertices.begin(), vertices.end(),
                                          [](SturGVertex const &lhs, SturGVertex const &rhs) {
                                              return lhs.vertex_param[2] < rhs.vertex_param[2];
                                          });
    temp_rectangle.min_vertices.z = minmax_vertices.first->vertex_param[2];
    temp_rectangle.max_vertices.z = minmax_vertices.second->vertex_param[2];

    // std::cout <<  "min: " << minmax_vertices.first->vertex_param[2] << " max: " <<
    // minmax_vertices.second->vertex_param[2] << std::endl;

    return temp_rectangle;
}

int RcLoader::getRandomColor(float rand_color_array[], uint64_t seed, unsigned int is_terrain) {
    // generate random for buildings from model files
    // the result is unchanged if its for cnn
    if (is_terrain) {
        rand_color_array[0] = 0.0f;
        rand_color_array[1] = 0.0f;
        rand_color_array[2] = 0.0f;
    } else {
        // input unique seed
        srand(static_cast<int>(seed * time(0)));
        // generate random numbers between 0 to 255;
        // making sure  R band is never zero
        // convert them to OpenGL colors float format
        rand_color_array[0] = (((rand() + 1) % 255) / 255.0f);
        rand_color_array[1] = (((rand() + 2) % 255) / 255.0f);
        rand_color_array[2] = (((rand() + 3) % 255) / 255.0f);

        // we are using red band. thus making sure its a non zero value
        // if(rand_color_array[0] == 0){
        //     rand_color_array[0] = (rand_color_array[1] + rand_color_array[2])/2.0f;
        // }
    }

    return 1;
}

int RcLoader::processDataforRendering() {
    int j;
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
                    // colors_.push_back(0.0);
                }

                else {
                    ter_vertices_.push_back(*vertice_iter);

                    for (j = 0; j < VERTEX_VERTICES_SIZE; j++) {
                        ter_colors_.push_back(rand_color_array[j]);
                        // To DO: needs fix if VERTEX_VERTICES_SIZE!=COLOR_PARAM_SIZE
                    }
                    // ter_colors_.push_back(0.0);
                }
            }
        }
        // std::cout << "vertices size now: " <<  vertices_.size() << std::endl;
    }

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

int RcLoader::displayVertices() {
    std::ofstream outputFile;
    outputFile.open("vertices.csv");
    outputFile.precision(8);

    std::cout << "vertices size :" << vertices_.size() << std::endl;
    for (auto pp_iter = vertices_.begin(); pp_iter != vertices_.end(); pp_iter++) {
        outputFile << pp_iter->vertex_param[0] << "," << pp_iter->vertex_param[1] << ","
                   << pp_iter->vertex_param[2] << std::endl;
    }
    outputFile.close();

    return 1;
}

// int RcLoader::displayMaxBindingRectangles() {

//     std::ofstream outputFile;
//     outputFile.open("mbrs.csv");
//     outputFile.precision(8);

//     std::cout << "mbr size :" << max_binding_rectangles_.size() << std::endl;
//     for (auto pp_iter = max_binding_rectangles_.begin(); pp_iter !=
//     max_binding_rectangles_.end(); pp_iter++) {
//         outputFile << pp_iter->x << "," << pp_iter->y << ","<< pp_iter->z <<"," << pp_iter->w <<
//         std::endl;
//     }
//     outputFile.close();

//     return 1;
// }

// RcLoader class destructor with input params
RcLoader::~RcLoader() {
    // at expense of speed
    std::vector<SturGBuildingData>().swap(buildings_);
    std::vector<uint64_t>().swap(tile_ids_);
    std::vector<SturGVertex>().swap(vertices_);
    std::vector<SturGVertex>().swap(ter_vertices_);
}
