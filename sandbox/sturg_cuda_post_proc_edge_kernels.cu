//
//  sturg_cuda_post_proc_edge_kernel.cu
//  sturgRender
//
//  Created by Dilip Patlolla on 11/18/18.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#ifndef _STURG_CUDA_POST_PROC_EDGE_KERNEL_
#define _STURG_CUDA_POST_PROC_EDGE_KERNEL_

#include <stdint.h>


#define BLOCK_SIZE_W 16
#define BLOCK_SIZE_H 16

// compute the edges or mask
// from input rgb raster, using only one band
//
// \param[in]  d_raster         input imagery raster rgb
// \param[in]  d_gradient_mag   computed image edges
// \param[out] d_scene_and      output computer from d_gradient_mad & d_ref_edges
// \param[out] d_scene_xor      output computer from d_gradient_mad & d_ref_edges
// \param[in]  width            raster width
// \param[in]  height           raster height
// \param[in] num_bands         the count of rgb i.e 3
// \param[in] edge              true if using edges else use mask
//

#define CLAMP(value, min, max) (((value) > (max)) ? (max) : (((value) < (min)) ? (min) : (value)))

__forceinline__ __device__ unsigned int hashRGB(float r, float g, float b) {
    unsigned int rgbHash = 0;
    rgbHash = ((uint8_t)(255.0f * CLAMP(r, 0.0f, 1.0f))) << 8;
    rgbHash += ((uint8_t)(255.0f * CLAMP(g, 0.0f, 1.0f))) << 16;
    rgbHash += ((uint8_t)(255.0f * CLAMP(b, 0.0f, 1.0f))) << 24;

    return rgbHash;
}

// TO DO: further optimize this raw implementation
// TO DO: optimize the access of shared mem to remove bank conflicts
__global__ void gpuGetGradAndOrientAng(const float* d_raster_rgb, unsigned char* d_gradient_mag,
                                       const int width, const int height, int num_bands,
                                       bool using_egde) {
    unsigned int tidx = threadIdx.x;
    unsigned int tidy = threadIdx.y;

    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + tidx;
    unsigned int y = blockIdx.y * blockDim.y + tidy;
    unsigned int xindex = y * width + x;
    unsigned int rgb_xindex = xindex * num_bands;
    float temp_mag = 0;

    if ((x < width) && (y < height)) {
        // Load ghost array required into shared mem
        // ghost pixels required to compute edges
        if (using_egde) {
            __shared__ unsigned int sh_mem_raster[BLOCK_SIZE_W][BLOCK_SIZE_H + 1];
            sh_mem_raster[tidx][tidy] =
                hashRGB(d_raster_rgb[rgb_xindex], d_raster_rgb[rgb_xindex + 1],
                        d_raster_rgb[rgb_xindex + 2]);
            
            __syncthreads();

            if (x == (width - 1)) {
                rgb_xindex = (y * width + x - 1) * num_bands;
                sh_mem_raster[tidx + 1][tidy] =
                    hashRGB(d_raster_rgb[rgb_xindex], d_raster_rgb[rgb_xindex + 1],
                            d_raster_rgb[rgb_xindex + 2]);
            }

            else if ((tidx == blockDim.x - 1)) {
                rgb_xindex = (y * width + x + 1) * num_bands;
                sh_mem_raster[tidx + 1][tidy] =
                    hashRGB(d_raster_rgb[rgb_xindex], d_raster_rgb[rgb_xindex + 1],
                            d_raster_rgb[rgb_xindex + 2]);
            }
            __syncthreads();

            if (y == (height - 1)) {
                rgb_xindex = ((y - 1) * width + x) * num_bands;
                sh_mem_raster[tidx][tidy + 1] =
                    hashRGB(d_raster_rgb[rgb_xindex], d_raster_rgb[rgb_xindex + 1],
                            d_raster_rgb[rgb_xindex + 2]);
            } else if ((tidy == blockDim.y - 1)) {
                rgb_xindex = ((y + 1) * width + x) * num_bands;
                sh_mem_raster[tidx][tidy + 1] =
                    hashRGB(d_raster_rgb[rgb_xindex], d_raster_rgb[rgb_xindex + 1],
                            d_raster_rgb[rgb_xindex + 2]);
            }

            __syncthreads();

            float temp_horz = sh_mem_raster[tidx + 1][tidy] - sh_mem_raster[tidx][tidy];
            float temp_vert = sh_mem_raster[tidx][tidy + 1] - sh_mem_raster[tidx][tidy];

            // not computing true magnitude. we need to know if its greater than zero or not
            temp_mag = (temp_horz * temp_horz + temp_vert * temp_vert);
            d_gradient_mag[xindex] = ((temp_mag) > 0.0) ? (unsigned char)255 : (unsigned char)0;
        } else {
            temp_mag = ((d_raster_rgb[rgb_xindex] + d_raster_rgb[rgb_xindex + 1] +
                         d_raster_rgb[rgb_xindex + 2]) > 0.0)? 1: 0;
            d_gradient_mag[xindex] = (temp_mag > 0.0) ? (unsigned char)1 : (unsigned char)0;
        }
    }
    __syncthreads();
}

// TO DO: further optimize this raw implementation
// TO DO: optimize the access of shared mem to remove bank conflicts
// overlay layers in bgr
//
// \param[in] d_raster_rgb  input imagery raster rgb
// \param[in] d_edges       computed image edges
// \param[in] width         raster width
// \param[in] height        raster height
// \param[in] num_bands     the count of rgb i.e 3
//

__global__ void gpuOverlayLayers(float4* d_raster_rgb, unsigned char* d_edges, const int width,
                                 const int height) {
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int xindex = y * width + x;
    unsigned int rgb_xindex = xindex;

    if ((x < width) && (y < height)) {
        float r = d_raster_rgb[rgb_xindex].x;
        float g = d_raster_rgb[rgb_xindex].y;
        float b = d_raster_rgb[rgb_xindex].z;

        if ((r == g) && (g == b) && (r > 0)) {
            r = 128;
            g = 128;
            b = 128;
        } else if (r == 0 && g == 0 && b == 0) {
            r = 0;
            g = 0;
            b = 0;
        } else {
            r = 255;
            g = 255;
            b = 255;
        }

        // check if edge layer has a valid pixel
        if (d_edges[xindex] == 255) {
            r = 255;
            g = 0;
            b = 0;
        }
        __syncthreads();

        // bgr conversion
        d_raster_rgb[rgb_xindex].x = b;
        d_raster_rgb[rgb_xindex].y = g;
        d_raster_rgb[rgb_xindex].z = r;
    }
    __syncthreads();
}

#endif  // #ifndef _STURG_CUDA_POST_PROC_EDGE_KERNEL_
