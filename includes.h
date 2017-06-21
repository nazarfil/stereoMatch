#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h> 
#include "iostream"
#include <stdio.h>
#include <time.h>
#include <windows.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


#define SIZEX 1920
#define SIZEY 1080
#define CHANNEL 3
#define DISPARITY_RANGE 7
#define BLOCK_SIZE 32
#define GRID_SIZE 4
#define TILE_DIM 32
#define BLOCK_ROW 8