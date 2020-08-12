//
// Created by yuqiong on 7/14/20.
//
// Include standard headers

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>


#include <iostream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "sturg_cuda_post_proc_edge_kernels.cu"
#include "sturg_cuda_post_proc_kernels.cu"

#define BLOCK_SIZE_W 16
#define BLOCK_SIZE_H 16

#define REDUCE_BLOCK_SIZE 64
#define REDUCE_THREAD_SIZE 256

#define STITCH_BLOCK_SIZE_W 36
#define STITCH_BLOCK_SIZE_H 36

#define ROI_SIZE_W 36
#define ROI_SIZE_H 36

#define DIVUP(a, b) (((a) % (b) != 0) ? (((a) + (b)-1) / (b)) : ((a) / (b)))

#define cudaErrorCheck(ans) \
    { cudaAssert((ans), __FILE__, __LINE__); }


inline void cudaAssert(cudaError_t err, const char* file_name, int line_num,
                       bool exit_on_err = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(err), file_name, line_num);
        if (exit_on_err) exit(err);
    }
}


int main() {
//    std::string img_path = "/media/yuqiong/DATA/ogl_sandbox/sandbox/build/buildings.png";
//    std::string img_path = "/media/yuqiong/DATA/ogl_sandbox/sandbox/build/cube.jpg";
    std::string img_path = "/media/yuqiong/DATA/ogl_sandbox/sandbox/build/imgcomp-440x330.png";

    cv::Mat image;
    image = cv::imread(img_path, cv::IMREAD_UNCHANGED);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", image );                   // Show our image inside it.

    cv::waitKey(0);                                          // Wait for a keystroke in the window

    int image_height= image.rows;
    int image_width = image.cols;
    int num_channels = image.channels();

    int num_pixels = image_height * image_width;
    int num_elements = num_pixels * num_channels;

    /************************* CPU data structures **************************/

    // convert image to float array
    std::vector<float> h_image(num_elements, 0);

    for (int i = 0; i < num_elements; i++) {
        h_image[i] = static_cast<float>(image.data[i]);
    }

    unsigned char *h_edges = new unsigned char[num_pixels]();

    float3 *h_surface_norm = new float3[num_pixels]();

    int image_size = num_elements * sizeof(float);
    int surface_norm_size = num_pixels * sizeof(float3);

    /************************* gpu data structures **************************/
    float* d_image;
    cudaErrorCheck(cudaMalloc(
            (void**)&d_image,
            image_size));

    float3* d_surface_norm;
    cudaErrorCheck(cudaMalloc(
            (void**)&d_surface_norm,
            surface_norm_size));

//    unsigned char* d_ref_edges_or_mask;
//    cudaErrorCheck(
//            cudaMalloc((void**)&d_ref_edges_or_mask,
//                       image_width * image_height * sizeof(char)));

    // allocate cuda device memory for gradient and orientation angle
//    unsigned char* d_edges_or_mask;
//    cudaErrorCheck(cudaMalloc(
//            (void**)&d_edges_or_mask,
//            image_width * image_height * sizeof(unsigned char)));

    /***************** run cuda kernel gradients *****************/

    cudaMemcpy(d_image, h_image.data(), image_size, cudaMemcpyHostToDevice);

    dim3 grid_dim(DIVUP(image_width, BLOCK_SIZE_W), DIVUP(image_height, BLOCK_SIZE_H), 1);
    dim3 block_dim(BLOCK_SIZE_W, BLOCK_SIZE_H, 1);

    // compute the gradient and orientation angle
//    gpuGetGradAndOrientAng<<<grid_dim, block_dim>>>(d_image, d_edges_or_mask,
//            image_width, image_height, num_channels, true);
//
//    cudaMemcpy(h_edges, d_edges_or_mask, out_size, cudaMemcpyDeviceToHost);

    // show outputs

//    cv::Mat out = cv::Mat(image_height, image_width, CV_8UC1);
//    memcpy(out.data, h_edges, out_size);
//
//    cv::imwrite("res.jpg", out);

//    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
//    cv::imshow( "Display window", out);                   // Show our image inside it.
//
//    cv::waitKey(0);                                          // Wait for a keystroke in the window

    /***************** run cuda kernel normal *****************/

    gpuGetUpdatedSurfaceNorm<<<grid_dim, block_dim>>>(d_image, d_surface_norm,
            image_width, image_height, num_channels);


    /***************** visualize results *****************/
    cv::Mat a, bgr;

    // download surface normal
    cv::cuda::GpuMat cv_image_surf(image_height, image_width, CV_32FC3, d_surface_norm);
    cv_image_surf.download(a);

    // download rgb data

    cv::cuda::GpuMat cv_image_rgb(image_height, image_width, CV_32FC4, d_image);
    cv_image_rgb.download(a);
    cv::cvtColor(a, bgr, cv::COLOR_BGRA2BGR);

//    cv::imwrite("res.jpg", a);

    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Display window", bgr);

    cv::waitKey(0);

    return 0;
}

