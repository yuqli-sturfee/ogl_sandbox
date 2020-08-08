//
// Created by yuqiong on 7/14/20.
//
// Understand cam_search_parameters

#include <iostream>
#include <string>
#include <vector>
#include <limits>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include "sturg_helper_func.hpp"
#include "sturg_loader.hpp"
#include "sturg_search_params.hpp"


int main() {
    //////////////////////////////////////////////////////////////////////////////////////////
    // Initialize parameters
    //////////////////////////////////////////////////////////////////////////////////////////

    // load config file
    auto config_params = loadConfigFile("config.ini");

    // read command line params
    SturgInputParams input_params;
    input_params.csv_file_path = "/media/yuqiong/DATA/ogl_sandbox/assets/test_data/extern_param_single.csv";
    input_params.proj_matrix_csv_file = "/media/yuqiong/DATA/ogl_sandbox/assets/test_data/proj_matrix_portrait.csv";
    input_params.window_origins_file = "/media/yuqiong/DATA/ogl_sandbox/assets/test_data/windows.csv";
    input_params.center_x = 553096.755;
    input_params.center_y = 4183086.188;
    input_params.fov = 45;
    input_params.scene_width = 640;
    input_params.scene_height = 360;
    input_params.utm_prefix = "10N";
    input_params.radius = 500;
    input_params.image_width = 640;
    input_params.image_height = 360;

    // update input_params with config init data
    input_params.model_dir = config_params["geometry_models"];
    input_params.terrain_dir = config_params["geometry_terrain"];

    std::cout << "Using: " << std::endl;
    std::cout << " > Geometry Models Dir: " << input_params.model_dir << std::endl;
    std::cout << " > Geometry Terrain Dir: " << input_params.terrain_dir << std::endl;

    std::cout << " > Win Origins File: " << input_params.window_origins_file << std::endl;
    std::cout << " > utm prefix: " << input_params.utm_prefix << std::endl;

    // parser
    sturgSearchParamData searchParamData;
    searchParamData.init(input_params);
    searchParamData.process();

    // get search params and other constsnt params for the scene: w,h,r etc
    std::vector<SturgCameraParameters> cam_search_params = searchParamData.getSearchParams();

    // TO DO: do not proceed of cam params size is zero
    if (cam_search_params.size() == 0) {
        std::cout << "Error: filtered cam params count is zero . exiting()";
        return EXIT_FAILURE;
    }

    std::cout << "Camera search param size ... " << cam_search_params.size() << std::endl;

    /*
    // print out camera search param vector
    for (int i = 0; i < cam_search_params.size(); i++) {
        auto cam_param = cam_search_params.at(i);
        std::cout << cam_param.param_name << std::endl;
        std::cout << "Camera parameters...\n";
        std::cout << cam_param.cam_x << " " << cam_param.cam_y << " " << cam_param.cam_z << std::endl;
        std::cout << "Rotation parameters...\n";
        std::cout << cam_param.yaw << " " << cam_param.pitch << " " << cam_param.roll << std::endl;
    }
    */



    std::vector<float> proj_matrix = searchParamData.getProjMatrix();

    // inspect cam_search_parameters

    // iterator to cam_search_params to extract name,yaw,pitch ..etc
    auto iter = cam_search_params.begin();


    // For speed computation
    time_t last_time_;
    time(&last_time_);
    double last_time = static_cast<double>(last_time_);
    int frame_count = 0;
    // first 2 [ i.e frame # 0 and # 1] frames we do not read data back to cpu
    // as pixel buffers are filled with valid data during these frames.
    int skip_frame_count = 2;  // skip frame no 0 and frame no 1 and then move to render next scene
    unsigned int yaw_count = 0;
    unsigned int pitch_count = 0;


    std::vector<float> scores(cam_search_params.size());

    // iterator to save max cam param
    auto max_iter = cam_search_params.begin();

    glm::mat4 View;

    View = glm::rotate(glm::mat4(1.0f), (glm::mediump_float)glm::radians(iter->pitch),
                       glm::vec3(-1.0f, 0.0f, 0.0f));
    View = glm::rotate(View, (glm::mediump_float)glm::radians(iter->roll),
                       glm::vec3(0.0f, -1.0f, 0.0f));
    View = glm::rotate(View, (glm::mediump_float)glm::radians(iter->yaw),
                       glm::vec3(0.0f, 0.0f, -1.0f));


    float center_x = 553096.755;
    float center_y = 4183086.188;

    // translate
    View = glm::translate(View, glm::vec3(-(iter->cam_x - center_x), -(iter->cam_y - center_y), -(iter->cam_z)));

    std::cout.precision(16);
        if (frame_count < cam_search_params.size() + skip_frame_count - 1) {
            std::cout << "param: " << iter->param_name << ", cam_x: " << iter->cam_x
                      << ", cam_y: " << iter->cam_y << ", "
                      << "center_x_" << center_x << ", "
                      << "center_y_" << center_y << std::endl;
            std::cout << "translate params: " << iter->cam_x - center_x << ","
                      << iter->cam_y - center_y << "," << iter->cam_z
                      << std::endl;
        }
    // Increment iter after first frame has been rendered
    if (iter != cam_search_params.end()) iter++;


}

