//
//  sturg_helper_func.cpp
//  sturfeeLoader
//
//  Created by Dilip Patlolla on 2/3/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#include "sturg_helper_func.hpp"

void sturgParseCmdParams(int argc, char** argv, SturgInputParams* cmdLineData) {
    int opt;
    bool args_provided = FALSE;
    std::string cmd_line_flags = "f:w:t:g:h:l:m:r:v:o:i:p:c:e";

#ifdef CNN
    cmd_line_flags = "f:u:w:t:g:h:l:m:r:v:o:c:e:";
#endif

#if defined(CNN) && defined(CAFFE_OUT)
    cmd_line_flags = "f:u:w:t:g:h:l:m:r:v:o:c:s:a:b:i:p:e";
#endif

    cmdLineData->use_edge = false;
    // get input command line parameters
    while ((opt = getopt(argc, argv, cmd_line_flags.c_str())) != EOF) {
        args_provided = TRUE;
        switch (opt) {
            // TO DO : make changes to the flags to make them meaningful
            case 'f':
                cmdLineData->csv_file_path = optarg;
                break;
            case 'a':
                cmdLineData->pca_mean_file_path = optarg;
                break;
            case 'b':
                cmdLineData->pca_coeff_file_path = optarg;
                break;
            case 't':
                cmdLineData->center_x = std::stod(optarg);
                break;
            case 'g':
                cmdLineData->center_y = std::stod(optarg);
                break;
            case 'w':
                cmdLineData->scene_width = atoi(optarg);
                break;
            case 'h':
                cmdLineData->scene_height = atoi(optarg);
                break;
            case 'r':
                cmdLineData->radius = atoi(optarg);
                break;
            case 'p':
                cmdLineData->proj_matrix_csv_file = optarg;
                break;
            case 'u':
                cmdLineData->utm_prefix = optarg;
                break;
            case 'v':
                cmdLineData->fov = atoi(optarg);
                break;
            case 'l':
                cmdLineData->image_width = atoi(optarg);
                break;
            case 'm':
                cmdLineData->image_height = atoi(optarg);
                break;
            case 'c':
                cmdLineData->cam_height = std::stod(optarg);
                break;

#if defined(CNN) && defined(CAFFE_OUT)
            case 'i':
                cmdLineData->image_ref_edge = optarg;
                checkFileExists(cmdLineData->image_ref_edge);
                break;
#endif
            case 'o':
                cmdLineData->output_dir = optarg;
                cmdLineData->output_dir += "/";
                cmdLineData->write_output = true;
                checkDirAndCreate(cmdLineData->output_dir);
                break;
            case 'e':
                cmdLineData->use_edge = true;
                break;
            case 's':
                cmdLineData->window_origins_file = optarg;
                break;

            default:
                sturgHelp();
                break;
        }
    }

    if (!cmdLineData->proj_matrix_csv_file.empty()) {
        checkFileExists(cmdLineData->proj_matrix_csv_file);
        checkCsvExtension(cmdLineData->proj_matrix_csv_file);
    }
    if (!cmdLineData->csv_file_path.empty()) {
        checkFileExists(cmdLineData->csv_file_path);
        checkCsvExtension(cmdLineData->csv_file_path);
    }

//    if (cmdLineData->utm_prefix.empty()) {
//        std::cout << "Error : Missing utm prefix" << std::endl;
//        sturgHelp();
//    }
#ifdef CAFFE_OUT
    if (!cmdLineData->window_origins_file.empty()) {
        checkFileExists(cmdLineData->window_origins_file);
        checkCsvExtension(cmdLineData->window_origins_file);
    }
#endif

#ifndef CNN
    checkFileExists(cmdLineData->image_ref_edge);
#else
// cmdLineData->use_edge   = TRUE;
#endif

    if (!args_provided) {
        std::cout << "Error : Missing input parameters" << std::endl;
        sturgHelp();
    }
}

void sturgHelp() {
    std::cout << "sturg Render help" << std::endl;
    std::cout << "Usage:   "
              << "./sturgRender"
              << " [-option] [arg]" << std::endl;
    std::cout << "option:  "
              << "-f  path/to/input/csv/file (required)" << std::endl;
    std::cout << "         "
              << "-t  curently camX (required)" << std::endl;
    std::cout << "         "
              << "-g  curently camY (required)" << std::endl;
    std::cout << "         "
              << "-u  utm prefix (required)" << std::endl;
    std::cout << "         "
              << "-w  scene width (required)" << std::endl;
    std::cout << "         "
              << "-h  scene height (required)" << std::endl;
    std::cout << "         "
              << "-l  output image width (default: (scene width/4)) (optional)" << std::endl;
    std::cout << "         "
              << "-m  output image height (default: (scene height/4)) (optional)" << std::endl;
    std::cout << "         "
              << "-r  radius (required)" << std::endl;
    std::cout << "         "
              << "-e  use edges (optional) (default (use mask))" << std::endl;
    std::cout << "         "
              << "-p  use .csv proj matrix param file (optional) (default (use glm perspective))"
              << std::endl;
#ifndef CNN
    std::cout << "         "
              << "-i  input image with reference edges(gray scale .png only) (required)"
              << std::endl;
#endif

#if defined(CNN) && defined(CAFFE_OUT)
    std::cout << "         "
              << "-i  input image for featuer descriptor (optional)" << std::endl;
#endif
    std::cout << "         "
              << "-o  output dir (optional)" << std::endl;
    std::cout << "         "
              << "-v  verbose (currently inactive)" << std::endl;
    exit(EXIT_FAILURE);
}

void checkFileExists(std::string input_file) {
    struct stat buf;
    //-check if file exists
    if (!(stat(input_file.c_str(), &buf) == 0)) {
        std::cout << std::endl
                  << "ERROR: input file " << input_file << " does not exist.\n",
            sturgHelp();
        exit(EXIT_FAILURE);
    }
}

void checkCsvExtension(std::string file_name) {
    if (!(file_name.substr(file_name.find_last_of(".") + 1) == "csv")) {
        std::cout << std::endl
                  << "ERROR: input file " << file_name << " needs to be .csv.\n",
            sturgHelp();
        exit(EXIT_FAILURE);
    }
}

void checkDirAndCreate(std::string output_dir) {
    struct stat st = {0};
    //-check for valid output dir
    if (!output_dir.empty()) {
        if (stat(output_dir.c_str(), &st) == -1) {
            printf("\n Output Dir %s does not exist. \t Creating Dir for output\n",
                   output_dir.c_str());
            mkdir(output_dir.c_str(), 0700);
        }
    }
}

// loads the configuration file -> config.ini
std::map<std::string, std::string> loadConfigFile(std::string filename) {
    // sample config file format
    /*
    geometry_models= "/home/ubuntu/models/"
    geometry_terrain= "/home/ubuntu/terrain_folder/"
    train_file= "/home/xyz/abc.caffemodel"
    model_file= "/home/xyz/lmn.prototxt"
    */

    std::ifstream input(filename);

    // A map of key-value pairs in the file
    std::map<std::string, std::string> config;

    // iterate over all the lines in the file
    while (input) {
        std::string key;
        std::string value;

        // Read up to the : delimiter into key
        std::getline(input, key, '=');
        // Read up to the newline into value
        std::getline(input, value, '\n');

        std::string::size_type pos1 =
            value.find_first_of("\"");  // Find the first quote in the value
        std::string::size_type pos2 = value.find_last_of("\"");  // Find the last quote in the value
        if (pos1 != std::string::npos && pos2 != std::string::npos &&
            pos2 > pos1)  // Check if the found positions are all valid
        {
            value = value.substr(
                pos1 + 1, pos2 - pos1 - 1);  // Take a substring of the part between the quotes
            config[key] = value;             // Store the result in the map
        }
    }
    input.close();  // Close the file stream

    // verify if the required values are defined in the config file
    if ((config.count("geometry_models") == 0) || (config.count("geometry_terrain") == 0) ||
        (config.count("caffe_train_file") == 0) || (config.count("caffe_model_file") == 0)) {
        std::cout << "ERROR > invalid config file. Exiting()" << std::endl;
        exit(EXIT_FAILURE);
    }

    return config;
}
