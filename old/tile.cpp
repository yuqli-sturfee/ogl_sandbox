// 2020-07-27
// Test loading individual tile and building data.
// Figure out the coordinate transforms

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>

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

float center_x_ = 0;
float center_y_ = 0;

struct SturGTileMetaData {
    uint32_t version, meta_length;
    uint32_t models_count;
    uint32_t tile_center_x, tile_center_y, tile_center_z;
};


struct SturGVertex {
    float vertex_param[VERTEX_VERTICES_SIZE] = {0, 0, 0};
    bool operator<(const SturGVertex that) const {
        return std::memcmp((void*)this, (void*)&that, sizeof(SturGVertex)) > 0;
    };
};


struct SturGFace {
    uint16_t face_vertex[FACE_VERTICES_SIZE] = {0, 0, 0};
};

int displayBinaryFileMeta(const SturGTileMetaData &bin_file_model_info) {
    std::cout << "meta_length :" << bin_file_model_info.meta_length << std::endl;
    std::cout << "models_count :" << bin_file_model_info.models_count << std::endl;
    std::cout << "tile_center_x :" << bin_file_model_info.tile_center_x << std::endl;
    std::cout << "tile_center_y :" << bin_file_model_info.tile_center_y << std::endl;

    return 1;
}

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

void displayBuildingsData(const std::vector<SturGBuildingData> & buildings_) {
    for (const SturGBuildingData &building_iter : buildings_) {
        std::cout << "Building ID: " << building_iter.id << "\t";
        std::cout << "No. of vertices: " << building_iter.vertices.size() << "\t";
        std::cout << "No. of faces: " << building_iter.faces.size() << std::endl;
        std::cout << "UINT16: " << building_iter.is_uint_16;
        std::cout << "\tVertices Byte Length: " << building_iter.vertices_byte_length;
        std::cout << "\tFaces Byte Length: " << building_iter.faces_byte_length << std::endl;
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
    }
}

std::vector<float> readSturgBinFile(const std::string file_name, unsigned int is_terrain) {
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

        displayBinaryFileMeta(bin_file_meta);

        std::cout << "Number of buildings " << buildings.size() << std::endl;

//        displayBuildingsData(buildings);

        // prepare data for rendering
        std::vector<float> vertices;

        // display a single building
//        for (int i = 0; i < 3; i++) {
        for (auto b : buildings) {
            std::cout << "New building !!!\n";
            // get vertices
            for (auto f : b.faces) {
                for (int i = 0; i < 3; i++) {
                    auto curr_vertex = b.vertices.at(f.face_vertex[i]);
                    vertices.push_back(curr_vertex.vertex_param[0] - bin_file_meta.tile_center_x);  // x
                    vertices.push_back(curr_vertex.vertex_param[1] - bin_file_meta.tile_center_y);  // y
                    vertices.push_back(curr_vertex.vertex_param[2] - bin_file_meta.tile_center_z);  // z
                    std::cout << "Vertex " << curr_vertex.vertex_param[0] - bin_file_meta.tile_center_x << " ";
                    std::cout << "Vertex " << curr_vertex.vertex_param[1] - bin_file_meta.tile_center_y << " ";
                    std::cout << "Vertex " << curr_vertex.vertex_param[2] - bin_file_meta.tile_center_z << " ";
                    std::cout << std::endl;
                }
            }
            std::cout << "===============================\n";
        }

//        std::cout << vertices.size() << std::endl;
//        for (auto v : vertices) {
//            std::cout << v << " ";
//        }
//        std::cout << std::endl;

        return vertices;
    }
    // if not handling this file...
    else {
        std::cout << " >>> skipping file:\t" << file_name << std::endl;
        return {};
    }

}



/*
int main() {
    std::string path = "/media/yuqiong/DATA/ogl_sandbox/data/sample/geometry_data/10N11140146925200";
    readSturgBinFile(path, 0);

    return 0;
}

 */