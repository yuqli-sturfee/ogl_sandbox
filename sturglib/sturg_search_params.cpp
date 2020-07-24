//
//  sturg_search_params.cpp
//  sturgRender
//
//  Created by Dilip Patlolla on 2/5/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

// MAJOR TO DO: remome duplicate functions and create a single genereic
// func for the csv reads and stores
#include "sturg_search_params.hpp"

sturgSearchParamData::sturgSearchParamData() {
    scene_width_ = 0;
    scene_height_ = 0;
    radius_ = 0;
    fov_ = 0;
    file_name_ = "";
}

int sturgSearchParamData::init(SturgInputParams input_data) {
    scene_width_ = input_data.scene_width;
    scene_height_ = input_data.scene_height;
    radius_ = input_data.radius;
    fov_ = input_data.fov;
    cam_height_ = input_data.cam_height;
    file_name_ = input_data.csv_file_path;
    pca_coeff_file_name_ = input_data.pca_coeff_file_path;
    pca_mean_file_name_ = input_data.pca_mean_file_path;

#ifdef CAFFE_OUT
    window_origins_file_ = input_data.window_origins_file;
#endif

    if (!input_data.proj_matrix_csv_file.empty()) {
        proj_file_name_ = input_data.proj_matrix_csv_file;
    }

    return 1;
}

int sturgSearchParamData::checkCsvExtension(std::string file_name) {
    if (file_name_.substr(file_name.find_last_of(".") + 1) == "csv")
        return 1;
    else
        exit(EXIT_FAILURE);
}

int sturgSearchParamData::process() {
    sturgSearchParamsReadCsv();
    if (!proj_file_name_.empty()) sturgSearchParamsReadProjCsv();
    if (!pca_mean_file_name_.empty()) sturgPcaMeanParamsReadCsv();
    if (!pca_coeff_file_name_.empty()) sturgPcaCoeffParamsReadCsv();

#ifdef CAFFE_OUT
    sturgReadWindowOriginCsv();
#endif

    return 1;
}

int sturgSearchParamData::storeWindowOriginParam(const std::vector<std::string> temp_vector) {
    std::array<float, 2> temp;
    temp[0] = std::stof(temp_vector[0]);
    temp[1] = std::stof(temp_vector[1]);

    window_orig_params_.push_back(temp);
}

int sturgSearchParamData::sturgReadWindowOriginCsv() {
    // Open file
    std::ifstream csv_file(window_origins_file_, std::ios::in);
    std::string csv_row;

    while (!csv_file.eof()) {
        // Vector to store data from each line
        std::vector<std::string> temp_vector;
        std::string tok;
        getline(csv_file, csv_row, '\n');
        if (csv_row == "") {
            continue;
        }
        // Turn the std::string into a stream.
        std::stringstream ss(csv_row);
        // Read the token with "," delimiter
        while (getline(ss, tok, ',')) {
            std::stringstream ss_temp(tok);
            temp_vector.push_back(tok);
        }
        if (temp_vector.size() < 2) {  // need 4 params per row
            std::cout << "corrupt input csv.\tminimum 4 parameters required/line" << std::endl;
            exit(EXIT_FAILURE);
        }
        // Store one seach parameter into search_params_
        this->storeWindowOriginParam(temp_vector);
    }
    csv_file.close();

    return 1;
}

// TO DO: duplicate funcs . cleanup and create a single func for read
int sturgSearchParamData::sturgPcaMeanParamsReadCsv() {
    pca_mean_params_.clear();

    // Open file
    std::ifstream csv_file(pca_mean_file_name_, std::ios::in);
    std::string csv_row;

    // Check if file exists
    if (!csv_file.good()) {
        std::cout << "ERROR : missing search param's csv file:\t" << pca_mean_file_name_
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    while (!csv_file.eof()) {
        // Vector to store data from each line
        std::vector<std::string> temp_vector;
        std::string tok;
        getline(csv_file, csv_row, '\n');
        if (csv_row == "") {
            continue;
        }
        // Turn the std::string into a stream.
        std::stringstream ss(csv_row);
        // Read the token with "," delimiter
        while (getline(ss, tok, ',')) {
            std::stringstream ss_temp(tok);
            temp_vector.push_back(tok);
        }
        if (temp_vector.size() < 1) {  // #define 6 min params/line in csv
            std::cout << "corrupt input csv.\tminimum 6 parameters required/line" << std::endl;
            exit(EXIT_FAILURE);
        }
        // Store one seach parameter into search_params_
        this->storePcaMeanParam(temp_vector);
    }
    csv_file.close();
    std::cout << "helper mean params size: " << pca_mean_params_.size() << std::endl;

    return 1;
}

// TO DO: duplicate funcs . cleanup and create a single func for read
int sturgSearchParamData::sturgPcaCoeffParamsReadCsv() {
    pca_coeff_params_.clear();

    // Open file
    std::ifstream csv_file(pca_coeff_file_name_, std::ios::in);
    std::string csv_row;

    // Check if file exists
    if (!csv_file.good()) {
        std::cout << "ERROR : missing search param's csv file:\t" << pca_coeff_file_name_
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    while (!csv_file.eof()) {
        // Vector to store data from each line
        std::vector<std::string> temp_vector;
        std::string tok;
        getline(csv_file, csv_row, '\n');
        if (csv_row == "") {
            continue;
        }
        // Turn the std::string into a stream.
        std::stringstream ss(csv_row);
        // Read the token with "," delimiter
        while (getline(ss, tok, ',')) {
            std::stringstream ss_temp(tok);
            temp_vector.push_back(tok);
        }
        if (temp_vector.size() < 6) {  // #define 6 min params/line in csv
            std::cout << "corrupt input csv.\tminimum 6 parameters required/line" << std::endl;
            exit(EXIT_FAILURE);
        }
        // Store one seach parameter into search_params_
        this->storePcaCoeffParam(temp_vector);
    }
    csv_file.close();

    return 1;
}

int sturgSearchParamData::storePcaMeanParam(const std::vector<std::string> temp_vector) {
    for (auto const& value : temp_vector) {
        pca_mean_params_.push_back(std::stof(value));
    }

    return 1;
}

int sturgSearchParamData::storePcaCoeffParam(const std::vector<std::string> temp_vector) {
    for (auto const& value : temp_vector) {
        pca_coeff_params_.push_back(std::stof(value));
    }

    return 1;
}

int sturgSearchParamData::sturgSearchParamsReadProjCsv() {
    // Open file
    std::ifstream csv_file(proj_file_name_, std::ios::in);
    std::string csv_row;

    while (!csv_file.eof()) {
        // Vector to store data from each line
        std::vector<std::string> temp_vector;
        std::string tok;
        getline(csv_file, csv_row, '\n');
        if (csv_row == "") {
            continue;
        }
        // Turn the std::string into a stream.
        std::stringstream ss(csv_row);
        // Read the token with "," delimiter
        while (getline(ss, tok, ',')) {
            std::stringstream ss_temp(tok);
            temp_vector.push_back(tok);
        }
        if (temp_vector.size() < 4) {  // need 4 params per row
            std::cout << "corrupt input csv.\tminimum 4 parameters required/line" << std::endl;
            exit(EXIT_FAILURE);
        }
        // Store one seach parameter into search_params_
        this->storeProjMatrixParam(temp_vector);
    }
    csv_file.close();

    return 1;
}

int sturgSearchParamData::sturgSearchParamsReadCsv() {
    // Check if input is a files with .csv extension
    // this->checkCsvExtension(file_name_);

    // Open file
    std::ifstream csv_file(file_name_, std::ios::in);
    std::string csv_row;

    // Check if file exists
    if (!csv_file.good()) {
        std::cout << "ERROR : missing search param's csv file:\t" << file_name_ << std::endl;
        exit(EXIT_FAILURE);
    }

    while (!csv_file.eof()) {
        // Vector to store data from each line
        std::vector<std::string> temp_vector;
        std::string tok;
        getline(csv_file, csv_row, '\n');
        if (csv_row == "") {
            continue;
        }
        // Turn the std::string into a stream.
        std::stringstream ss(csv_row);
        // Read the token with "," delimiter
        while (getline(ss, tok, ',')) {
            std::stringstream ss_temp(tok);
            temp_vector.push_back(tok);
        }
        if (temp_vector.size() < 6) {  // #define 6 min params/line in csv
            std::cout << "corrupt input csv.\tminimum 6 parameters required/line" << std::endl;
            exit(EXIT_FAILURE);
        }
        // Store one seach parameter into search_params_
        this->storeSearchParam(temp_vector);
    }
    csv_file.close();

    return 1;
}

int sturgSearchParamData::storeProjMatrixParam(const std::vector<std::string> temp_vector) {
    proj_matrix_params_.push_back(std::stof(temp_vector[0]));
    proj_matrix_params_.push_back(std::stof(temp_vector[1]));
    proj_matrix_params_.push_back(std::stof(temp_vector[2]));
    proj_matrix_params_.push_back(std::stof(temp_vector[3]));

    return 1;
}

int sturgSearchParamData::storeSearchParam(const std::vector<std::string> temp_vector) {
    SturgCameraParameters tempParams;

    // convert from string to respective datatypes

    tempParams.param_name = temp_vector[0];
    tempParams.cam_x = std::stod(temp_vector[1]);
    tempParams.cam_y = std::stod(temp_vector[2]);
    tempParams.cam_z = std::stod(temp_vector[3]);
    tempParams.yaw = std::stof(temp_vector[4]);
    tempParams.pitch = std::stof(temp_vector[5]);
    tempParams.roll = std::stof(temp_vector[6]);
// tempParams.fov   = std::stof(temp_vector[7]);

#ifdef DEBUG_RENDERS
    search_params_.push_back(tempParams);
#else
    // compute the params for yaw range and add to the cam params vector
    for (auto i = 0; i < NUM_OF_YAW_STEPS; i++) {
        // Store one cam param parameter into search_params_
        tempParams.yaw = yawStepsTable[i];
        for (auto j = 0; j < NUM_OF_PITCH_STEPS; j++) {
            tempParams.pitch = pitchStepsTable[j];
            search_params_.push_back(tempParams);
            // this->storeSearchParam(temp_vector);
        }
    }
#endif

    return 1;
}
sturgSearchParamData::~sturgSearchParamData() {}

std::vector<SturgCameraParameters> sturgSearchParamData::getFilteredParams() {
    return spatial_filtered_camparams_;
}

std::vector<SturgCameraParameters> sturgSearchParamData::getSearchParams() {
    return search_params_;
}

std::vector<float> sturgSearchParamData::getProjMatrix() { return proj_matrix_params_; }

std::vector<float> sturgSearchParamData::getPcaMean() { return pca_mean_params_; }

std::vector<float> sturgSearchParamData::getPcaCoeff() { return pca_coeff_params_; }

std::vector<std::array<float, 2>> sturgSearchParamData::getWindowOrigins() {
    return window_orig_params_;
}

float sturgSearchParamData::getRadius() { return radius_; }
int sturgSearchParamData::getSceneWidth() { return scene_width_; }
int sturgSearchParamData::getSceneHeight() { return scene_height_; }
float sturgSearchParamData::getFov() { return fov_; }
