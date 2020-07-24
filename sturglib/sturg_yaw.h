//
//  sturg_yaw.h
//  predefined yaw steos
//
//  Created by Dilip Patlolla on 02/21/18.
//  Copyright (c) 2015-2025 STURFEE INC ALL RIGHTS RESERVED
//

#ifndef STURG_YAW_H
#define STURG_YAW_H

#define NUM_OF_YAW 207
#define NUM_OF_YAW_STEPS 104
#define NUM_OF_PITCH_STEPS 26

#define PATCH_DESCRIPTOR_SIZE 128
#define MIN_REDCUE_BLOCK_SIZE 1024

extern float yawTable[NUM_OF_YAW];
extern float yawStepsTable[NUM_OF_YAW_STEPS];
extern float pitchStepsTable[NUM_OF_PITCH_STEPS];

#endif /* STURG_YAW_H */
