//
//  sturg_helper_func.hpp
//  sturfeeLoader
//
//  Created by Dilip Patlolla on 2/3/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#ifndef sturg_helper_func_hpp
#define sturg_helper_func_hpp

#include <png.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <iterator>  // std:: distance
#include <list>      // std::list<>
#include <map>       // std::map
#include <vector>    // std::vector<>

#include "sturg_struct.h"
#define NUM_OF_COLOR_BANDS 3

#ifdef DEBUG1
#ifndef DEBUG
#define DEBUG
#endif
#endif

#ifdef CNN
#ifndef DEBUG
#define DEBUG
#endif
#endif

void sturgHelp();
std::map<std::string, std::string> loadConfigFile(std::string filename);
void sturgParseCmdParams(int argc, char** argv, SturgInputParams* cmdLineData);
void checkFileExists(std::string input_file);
void checkCsvExtension(std::string file_name);
void checkDirAndCreate(std::string output_dir);
int writePngOutput();

#endif /* sturg_helper_func_hpp */
