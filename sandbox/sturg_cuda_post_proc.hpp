//
//  sturg_cuda_post_proc.hpp
//  sturgRender
//
//  Created by Dilip Patlolla on 2/17/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#ifndef STURG_CUDA_POST_PROC_HPP
#define STURG_CUDA_POST_PROC_HPP

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <sys/time.h>
#include <algorithm>
#include <iostream>  // std::std::cout
#include <list>      // std::list
#include <vector>    // std::vector

#include <npp.h>

#include "helper/sturg_helper_func.hpp"
#include "helper/sturg_search_params.hpp"
#include "png/sturg_png.hpp"

#include "cnn/sturg_cnn.hpp"

#define STITCHED_IMAGE_HEIGHT 918
#define STITCHED_IMAGE_WIDTH 3726

#define BLOCK_SIZE_W 16
#define BLOCK_SIZE_H 16

#define REDUCE_BLOCK_SIZE 64
#define REDUCE_THREAD_SIZE 256

#define STITCH_BLOCK_SIZE_W 36
#define STITCH_BLOCK_SIZE_H 36

#define ROI_SIZE_W 36
#define ROI_SIZE_H 36

#define DIVUP(a, b) (((a) % (b) != 0) ? (((a) + (b)-1) / (b)) : ((a) / (b)))

typedef struct timestruct {
    unsigned int sec;
    unsigned int usec;

} TimeStruct;

// TO DO; move to sturg_time.h
TimeStruct getCurrentTime(void);

// TO DO; move to sturg_time.h
TimeStruct getCurrentTimeCPU(void);

double GetTimerValue(TimeStruct first, TimeStruct last);

int iDivUp(int a, int b);

unsigned char* sturgCudaPostProcess(float* image_raster, unsigned char* h_ref_edges,
                                    int image_width, int image_height, int num_bands);

float postProcessInLoop(float* d_image_raster, float3* d_surface_norm, float* d_stitched_image,
                        unsigned char* d_edges_or_mask, unsigned char* d_ref_edges_or_mask,
                        int num_bands, SturgInputParams cmd_line_data, Classifier& caffe_classifier,
                        unsigned int yaw_count, unsigned int pitch_count, std::string param_name);

#define cudaErrorCheck(ans) \
    { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t err, const char* file_name, int line_num,
                       bool exit_on_err = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(err), file_name, line_num);
        if (exit_on_err) exit(err);
    }
}

#endif /* STURG_CUDA_POST_PROC_HPP */
