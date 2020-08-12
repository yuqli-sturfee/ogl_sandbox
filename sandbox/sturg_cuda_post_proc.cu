//
//  sturg_cuda_post_proc.cpp
//  sturgRender
//
//  Created by Dilip Patlolla on 2/17/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#include "cuda/sturg_cuda_post_proc.hpp"
#include "cuda/sturg_cv_post_proc.hpp"
#include "render/sturg_render.hpp"

#include "cnn/sturg_cnn.hpp"
#include "sturg_cuda_post_proc_edge_kernels.cu"
#include "sturg_cuda_post_proc_kernels.cu"

// TO DO: make post process a class with cuda func
// TO DO: eliminate the need for cudaMemcpy()
float postProcessInLoop(float* d_image_raster, float3* d_surface_norm, float* d_stitched_image,
                        unsigned char* d_edges_or_mask, unsigned char* d_ref_edges_or_mask,
                        int num_bands, SturgInputParams sturg_input_data,
                        Classifier& caffe_classifier, unsigned int yaw_count,
                        unsigned int pitch_count, std::string param_name) {

    static const NppiSize npp_mask_size = {5, 5};
    static const NppiSize npp_roi = {sturg_input_data.image_width, sturg_input_data.image_height};
    static const NppiSize npp_dim = {sturg_input_data.image_width, sturg_input_data.image_height};
    static const NppiPoint npp_anchor = {2, 2};
    static const NppiPoint npp_offset = {0, 0};

    // allocate gpu memory to store dilation mask
    static unsigned char* npp_mask;
    static int count = 0;
    if (count == 0) {
        cudaErrorCheck(
            cudaMalloc(reinterpret_cast<void**>(&npp_mask),
                       sizeof(unsigned char) * npp_mask_size.height * npp_mask_size.width));
        cudaErrorCheck(cudaMemset(npp_mask, 1, npp_mask_size.height * npp_mask_size.width));
    }
    count++;

    // TODO: process only ROI rather than whole scene
    dim3 dimG_BASE(DIVUP(sturg_input_data.image_width, BLOCK_SIZE_W),
                   DIVUP(sturg_input_data.image_height, BLOCK_SIZE_H), 1);
    dim3 dimB_BASE(BLOCK_SIZE_W, BLOCK_SIZE_H, 1);

    // declare cuda grid and block dimensions
    dim3 dimG_STITCH(1, 2, 1);
    dim3 dimB_STITCH(ROI_SIZE_W, ROI_SIZE_H / 2, 1);

    // compute the gradient and orientation angle
    gpuGetGradAndOrientAng<<<dimG_BASE, dimB_BASE, 0>>>(
        d_image_raster, d_edges_or_mask, sturg_input_data.image_width,
        sturg_input_data.image_height, NUM_OF_OPENGL_OUTPUT_BANDS, sturg_input_data.use_edge);

    // get surface norm rendering
    gpuGetUpdatedSurfaceNorm<<<dimG_BASE, dimB_BASE, 0>>>(d_image_raster, d_surface_norm,
                                                          sturg_input_data.image_width,
                                                          sturg_input_data.image_height, num_bands);
    
    // exclude border and dilate
    nppiDilateBorder_8u_C1R(d_edges_or_mask, sturg_input_data.image_width, npp_dim, npp_offset,
                            d_ref_edges_or_mask, sturg_input_data.image_width, npp_roi, npp_mask,
                            npp_mask_size, npp_anchor, NPP_BORDER_REPLICATE);

    // call cuda kernel to compute the gradient and orientation angle
    gpuOverlayLayers<<<dimG_BASE, dimB_BASE, 0>>>((float4*)d_image_raster, d_ref_edges_or_mask,
                                                  sturg_input_data.image_width,
                                                  sturg_input_data.image_height);

#ifdef DEBUG_RENDERS
        cv::Mat bgr,a;

        cv::cuda::GpuMat cv_image_surf(sturg_input_data.image_height, sturg_input_data.image_width, CV_32FC3,
                                  d_surface_norm);
        cv_image_surf.download(a);

        cv::imwrite(sturg_input_data.output_dir + param_name + "_surf.png", a);

        cv::cuda::GpuMat cv_image_rgb(sturg_input_data.image_height, sturg_input_data.image_width, CV_32FC4,
                                  d_image_raster);
        cv_image_rgb.download(a);
        cv::cvtColor(a, bgr, cv::COLOR_BGRA2BGR);
        cv::imwrite(sturg_input_data.output_dir + param_name + "_edge.png", bgr);    
#else

#ifdef DEBUG3
        cv::Mat bgr,a;

        cv::cuda::GpuMat cv_image_surf(sturg_input_data.image_height, sturg_input_data.image_width, CV_32FC3,
                                  d_surface_norm);
        cv_image_surf.download(a);

        cv::resize(a, a, cv::Size(), 0.5, 0.5);

        cv::imwrite(sturg_input_data.output_dir + param_name + "_" + std::to_string(yaw_count) + "_" + std::to_string(pitch_count) + "_surf.png", a);


        cv::cuda::GpuMat cv_image_rgb(sturg_input_data.image_height, sturg_input_data.image_width, CV_32FC4,
                                  d_image_raster);
        cv_image_rgb.download(a);
        cv::cvtColor(a, bgr, cv::COLOR_BGRA2BGR);
        cv::resize(bgr, bgr, cv::Size(), 0.5, 0.5);
        cv::imwrite(sturg_input_data.output_dir + param_name + "_" + std::to_string(yaw_count) + "_" + std::to_string(pitch_count) + "_edge.png", bgr);     
#endif
    // stitch the global image 6 bands. 3 from edge and 3 from surface norm
    updateGlobalImage<<<dimG_STITCH, dimB_STITCH, 0>>>(
        d_surface_norm, (float4*)d_image_raster, d_stitched_image, yaw_count, pitch_count,
        sturg_input_data.image_width, sturg_input_data.image_height, NUM_OF_CNN_INPUT_BANDS);


    // process the fully stitched image to generate the descriptor
    if (yaw_count == NUM_OF_YAW_STEPS - 1 && pitch_count == NUM_OF_PITCH_STEPS - 1) {
        cudaDeviceSynchronize();
        caffe_classifier.processImage(d_stitched_image, STITCHED_IMAGE_WIDTH, STITCHED_IMAGE_HEIGHT,
                                      sturg_input_data.output_dir + "/" + param_name + ".bin");
        std::ofstream depth_result;
        depth_result.open(sturg_input_data.output_dir + "/" + param_name + ".done");
        depth_result.close();

#ifdef DEBUG2
        cv::cuda::GpuMat cv_image(STITCHED_IMAGE_HEIGHT, STITCHED_IMAGE_WIDTH * 6, CV_32FC1,
                                  d_stitched_image);
        cv::Mat a;
        cv_image.download(a);
        cv::imwrite(param_name + "_opencv.png", a);

       // std::ofstream myfile;
       // myfile.open(sturg_input_data.output_dir + "/" + param_name + "_opencv.csv");
       // myfile << cv::format(a, cv::Formatter::FMT_CSV) << std::endl;
       // myfile.close();
#endif

    }
#endif

    return 1;
}

TimeStruct getCurrentTime(void) {
    static struct timeval time_val;
    static struct timezone time_zone;

    TimeStruct time;

    cudaThreadSynchronize();
    gettimeofday(&time_val, &time_zone);

    time.sec = time_val.tv_sec;
    time.usec = time_val.tv_usec;
    return (time);
}

TimeStruct getCurrentTimeCPU(void) {
    static struct timeval time_val;
    static struct timezone time_zone;

    TimeStruct time;

    gettimeofday(&time_val, &time_zone);

    time.sec = time_val.tv_sec;
    time.usec = time_val.tv_usec;
    return (time);
}

double GetTimerValue(TimeStruct first, TimeStruct last) {
    int sec, usec;

    sec = last.sec - first.sec;
    usec = last.usec - first.usec;

    return (1000. * (double)(sec) + (double)(usec)*0.001);
}
