//
//  sturg_cuda_post_proc_kernel.cu
//  sturgRender
//
//  Created by Dilip Patlolla on 2/18/17.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#ifndef _STURG_CUDA_POST_PROC_KERNEL_
#define _STURG_CUDA_POST_PROC_KERNEL_

#define CONV_KERNEL_RADIUS 1

// compute the edges or mask
// from input rgb raster, using only one band
//
// \param[in]  d_raster         input imagery raster rgb
// \param[out] d_surface_norm   placeholder for surface normals
// \param[in]  width            raster width
// \param[in]  height           raster height
// \param[in] num_bands         the count of rgb i.e 3
//

// TO DO: further optimize this raw implementation
__global__ void gpuGetUpdatedSurfaceNorm(const float* d_raster, float3* d_surface_norm,
                                         const int width, const int height, int num_output_bands) {
    unsigned int tidx = threadIdx.x;
    unsigned int tidy = threadIdx.y;

    // for horizontal and vertical
    unsigned int x_tid_h;
    unsigned int y_tid_h;
    unsigned int x_tid_v;
    unsigned int y_tid_v;

    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + tidx;
    unsigned int y = blockIdx.y * blockDim.y + tidy;

    unsigned int xindex = (y * width + x) * 4;
    unsigned int surf_norm_xindex = y * width + x;
    unsigned int rgb_xindex = xindex;

    float norm = 0;
    float near = 0.01;
    float far = 2000;
    float eq_const_1 = (2 * near * far);
    float eq_const_2 = (far / 2.0 - near);
    float eq_const_3 = (far / 2.0 + near);
    float min = -1.0f;
    float max = 1.0f;

    __shared__ float sh_mem_raster_horz[BLOCK_SIZE_W + 2][BLOCK_SIZE_H];
    __shared__ float sh_mem_raster_vert[BLOCK_SIZE_W][BLOCK_SIZE_H + 2];

    if ((x < width) && (y < height)) {
        // Load ghost array required into shared mem
        // ghost pixels required to compute edges

        x_tid_h = threadIdx.x + 1 * CONV_KERNEL_RADIUS;
        y_tid_h = threadIdx.y;

        sh_mem_raster_horz[x_tid_h][y_tid_h] =
            (eq_const_1 / (eq_const_3 - d_raster[xindex + 3] * eq_const_2));
        
        __syncthreads();

        // load left Halo
        if (tidx == 0) {
            if (x == 0) {
                sh_mem_raster_horz[tidx][tidy] = sh_mem_raster_horz[x_tid_h][y_tid_h];
            } else {
                sh_mem_raster_horz[tidx][tidy] =
                    (eq_const_1 / (eq_const_3 - d_raster[xindex - 1 * 4 + 3] * eq_const_2));
            }
        }
        __syncthreads();

        // load right Halo
        if (x == (width - 1)) {
            sh_mem_raster_horz[tidx + 2 * CONV_KERNEL_RADIUS][tidy] =
                (eq_const_1 / (eq_const_3 - d_raster[xindex + 3] * eq_const_2));
        } else if ((tidx == blockDim.x - 1)) {
            rgb_xindex = xindex + 1 * 4 + 3;
            sh_mem_raster_horz[tidx + 2 * CONV_KERNEL_RADIUS][tidy] =
                (eq_const_1 / (eq_const_3 - d_raster[rgb_xindex] * eq_const_2));
        }

        __syncthreads();

        // update core part to vertical sh mem
        x_tid_v = threadIdx.x;
        y_tid_v = threadIdx.y + 1 * CONV_KERNEL_RADIUS;
        sh_mem_raster_vert[x_tid_v][y_tid_v] = sh_mem_raster_horz[x_tid_h][y_tid_h];

        __syncthreads();

        // load top halo
        if (tidy == 0) {
            if (y == 0) {
                sh_mem_raster_vert[tidx][tidy] = sh_mem_raster_vert[x_tid_v][y_tid_v];
            } else {
                sh_mem_raster_vert[tidx][tidy] =
                    (eq_const_1 /
                     (eq_const_3 - d_raster[((y - 1) * width + x) * 4 + 3] * eq_const_2));
            }
        }
        __syncthreads();

        // load bottom Halo
        if (y == (height - 1)) {
            sh_mem_raster_vert[tidx][tidy + 2 * CONV_KERNEL_RADIUS] =
                (eq_const_1 / (eq_const_3 - d_raster[xindex + 3] * eq_const_2));
        } else if ((tidy == blockDim.y - 1)) {
            rgb_xindex = ((y + 1) * width + x) * 4 + 3;
            sh_mem_raster_vert[tidx][tidy + 2 * CONV_KERNEL_RADIUS] =
                (eq_const_1 / (eq_const_3 - d_raster[rgb_xindex] * eq_const_2));
        }

        __syncthreads();

        // compute dx
        float temp_horz =
            (sh_mem_raster_horz[x_tid_h + 1][y_tid_h] - sh_mem_raster_horz[x_tid_h - 1][y_tid_h]) /
            2.0;

        // compute dy
        float temp_vert =
            (sh_mem_raster_vert[x_tid_v][y_tid_v + 1] - sh_mem_raster_vert[x_tid_v][y_tid_v - 1]) /
            2.0;

        // compute norm
        norm = sqrtf(temp_horz * temp_horz + temp_vert * temp_vert + 1.0f);

        if (d_raster[xindex + 3] == 0.0f) {
            d_surface_norm[surf_norm_xindex].z = 127;
            d_surface_norm[surf_norm_xindex].y = 127;
            d_surface_norm[surf_norm_xindex].x = 127;
        } else {
            // // version 1
            d_surface_norm[surf_norm_xindex].x = 255 * ((1.0f / norm) - min) / (max - min);
            d_surface_norm[surf_norm_xindex].y = 255 * ((-temp_horz / norm) - min) / (max - min);
            d_surface_norm[surf_norm_xindex].z = 255 * ((-temp_vert / norm) - min) / (max - min);
        }
    }
    __syncthreads();
}

// TO DO: further optimize this raw implementation
// TO DO: optimize the access of shared mem to remove bank conflicts
__global__ void scaleRaster(float3* d_raster, const double min, const double max, const int width,
                            const int height, int num_output_bands) {
    unsigned int tidx = threadIdx.x;
    unsigned int tidy = threadIdx.y;

    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + tidx;
    unsigned int y = blockIdx.y * blockDim.y + tidy;

    unsigned int xindex = y * width + x;

    if ((x < width) && (y < height)) {
        // Load ghost array required into shared mem
        // ghost pixels required to compute edges
        if (d_raster[xindex].x == 0.0f && d_raster[xindex].y == 0.0f &&
            d_raster[xindex].z == 0.0f) {
            d_raster[xindex].x = 127;
            d_raster[xindex].y = 127;
            d_raster[xindex].z = 127;
        } else if (min != max) {
            d_raster[xindex].x = 255 * (d_raster[xindex].x - min) / (max - min);
            d_raster[xindex].y = 255 * (d_raster[xindex].y - min) / (max - min);
            d_raster[xindex].z = 255 * (d_raster[xindex].z - min) / (max - min);
        } else {
            d_raster[xindex].x = 127;
            d_raster[xindex].y = 127;
            d_raster[xindex].z = 127;
        }
    }
    __syncthreads();
}
//
//// TO DO: further optimize this raw implementation
//// TO DO: optimize the access of shared mem to remove bank conflicts
//__global__ void updateGlobalImage(float3* d_surfnorm_raster, float4* d_image_raster,
//                                  float* d_global_raster, const unsigned int yaw_count,
//                                  const unsigned int pitch_count, const int width, const int height,
//                                  const int num_bands) {
//    unsigned int tidx = threadIdx.x;
//    unsigned int tidy = threadIdx.y;
//
//    // calculate normalized texture coordinates
//    unsigned int x = blockIdx.x * blockDim.x + tidx;
//    unsigned int y = blockIdx.y * blockDim.y + tidy;
//
//    unsigned int raster_xindex = y * width + x + width / 2 + height / 2 * width;
//    unsigned int global_xindex =
//        (y + pitch_count * 36) * STITCHED_IMAGE_WIDTH + x + yaw_count * 36 / 2;
//
//    // Load ghost array required into shared mem
//    // ghost pixels required to compute edges
//    if (pitch_count == 0) {
//        if (blockIdx.y == 1) {
//            if ((x < width) && (y < height) &&
//                ((y + (pitch_count - 0.5) * 36) < STITCHED_IMAGE_HEIGHT) &&
//                ((x + yaw_count * 36) < STITCHED_IMAGE_WIDTH)) {
//                global_xindex =
//                    (y + (pitch_count - 0.5) * 36) * STITCHED_IMAGE_WIDTH + x + yaw_count * 36;
//
//                d_global_raster[global_xindex * num_bands + 0] =
//                    d_image_raster[raster_xindex].x - B_MEAN;
//                d_global_raster[global_xindex * num_bands + 1] =
//                    d_image_raster[raster_xindex].y - G_MEAN;
//                d_global_raster[global_xindex * num_bands + 2] =
//                    d_image_raster[raster_xindex].z - R_MEAN;
//                d_global_raster[global_xindex * num_bands + 3] =
//                    d_surfnorm_raster[raster_xindex].x - B_MEAN;
//                d_global_raster[global_xindex * num_bands + 4] =
//                    d_surfnorm_raster[raster_xindex].y - G_MEAN;
//                d_global_raster[global_xindex * num_bands + 5] =
//                    d_surfnorm_raster[raster_xindex].z - R_MEAN;
//            }
//        }
//    } else {
//        if ((x < width) && (y < height) &&
//            ((y + (pitch_count - 0.5) * 36) < STITCHED_IMAGE_HEIGHT) &&
//            ((x + yaw_count * 36) < STITCHED_IMAGE_WIDTH)) {
//            global_xindex =
//                (y + (pitch_count - 0.5) * 36) * STITCHED_IMAGE_WIDTH + x + yaw_count * 36;
//            d_global_raster[global_xindex * num_bands + 0] =
//                d_image_raster[raster_xindex].x - B_MEAN;
//            d_global_raster[global_xindex * num_bands + 1] =
//                d_image_raster[raster_xindex].y - G_MEAN;
//            d_global_raster[global_xindex * num_bands + 2] =
//                d_image_raster[raster_xindex].z - R_MEAN;
//            d_global_raster[global_xindex * num_bands + 3] =
//                d_surfnorm_raster[raster_xindex].x - B_MEAN;
//            d_global_raster[global_xindex * num_bands + 4] =
//                d_surfnorm_raster[raster_xindex].y - G_MEAN;
//            d_global_raster[global_xindex * num_bands + 5] =
//                d_surfnorm_raster[raster_xindex].z - R_MEAN;
//        }
//    }
//    __syncthreads();
//}

#endif  // #ifndef _STURG_CUDA_POST_PROC_KERNEL_
