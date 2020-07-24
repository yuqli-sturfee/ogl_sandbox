//
//  sturg_search_params.hpp
//  sturgRender
//
//  Created by Dilip Patlolla on 2/5/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#ifndef sturg_search_params_hpp
#define sturg_search_params_hpp

#include <stdio.h>
#include <sys/stat.h>  // stat
#include <fstream>     // infile
#include <iostream>    // std::cout
#include <sstream>     // ss
#include <vector>      // std::vector<>

#include "point_in_triangle.hpp"
#include "rc_loader.hpp"
#include "sturg_struct.h"
#include "sturg_yaw.h"

class sturgSearchParamData : public RcLoader {
   public:
    // constructors
    sturgSearchParamData();

    int init(SturgInputParams input_data);

    int process();

    std::vector<SturgCameraParameters> getFilteredParams();
    std::vector<SturgCameraParameters> getSearchParams();
    std::vector<std::array<float, 2>> getWindowOrigins();
    std::vector<float> getProjMatrix();
    std::vector<float> getPcaCoeff();
    std::vector<float> getPcaMean();

    float getRadius();
    float getFov();
    int getSceneWidth();
    int getSceneHeight();
    int setSceneWidthAndHeight();

    // destructor
    ~sturgSearchParamData();

   private:
    std::string file_name_, proj_file_name_, window_origins_file_;
    std::string pca_coeff_file_name_, pca_mean_file_name_;

    float radius_, fov_;
    std::vector<SturgCameraParameters> spatial_filtered_camparams_;
    int scene_width_, scene_height_;
    double cam_height_;
    std::vector<SturgCameraParameters> search_params_;
    std::vector<float> proj_matrix_params_;
    std::vector<std::array<float, 2>> window_orig_params_;
    std::vector<float> pca_mean_params_;
    std::vector<float> pca_coeff_params_;

    int checkCsvExtension(std::string file_name);
    int storeSearchParam(const std::vector<std::string>);
    int storeProjMatrixParam(const std::vector<std::string>);
    int storeWindowOriginParam(const std::vector<std::string>);
    int storePcaCoeffParam(const std::vector<std::string> temp_vector);
    int storePcaMeanParam(const std::vector<std::string> temp_vector);
    int sturgReadWindowOriginCsv();
    int sturgSearchParamsReadCsv();
    int sturgSearchParamsReadProjCsv();
    int sturgPcaMeanParamsReadCsv();
    int sturgPcaCoeffParamsReadCsv();
};

#endif /* sturg_search_params_hpp */
