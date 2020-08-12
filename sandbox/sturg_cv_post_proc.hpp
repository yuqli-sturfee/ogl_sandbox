//
//  sturg_cv_post_proc.hpp
//  sturgRender
//
//  Created by Dilip Patlolla on 2/17/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#ifndef STURG_CV_POST_PROC_HPP
#define STURG_CV_POST_PROC_HPP

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <sys/time.h>
#include <algorithm>
#include <iostream>  // std::std::cout
#include <list>      // std::list
#include <vector>    // std::vector

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>

int preProcessPatch(cv::cuda::GpuMat& cv_image, std::vector<std::array<float, 2>> window_origins,
                    std::string param_file_name, unsigned int width, unsigned int height,
                    Classifier& caffe_classifier);

#endif /* STURG_CV_POST_PROC_HPP */
