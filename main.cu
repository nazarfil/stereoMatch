#include "includes.h"
#define SIZEX 1920
#define SIZEY 1080
#define CHANNEL 3
#define DISPARITY_RANGE 7
#define BLOCK_SIZE 32
#define GRID_SIZE 4
#define TILE_DIM 32
#define BLOCK_ROW 8
#include <stdlib.h> 
#include "iostream"
using namespace std;

void ToBlack(unsigned char* in, unsigned char* out, int x, int y);
void DeltaColor(unsigned char* left, unsigned char* right, unsigned char* out, int x, int y);
void GetGradient(unsigned char *img_left, unsigned char *img_right, unsigned char *out, int x, int y);
void GetCost(unsigned char *color, unsigned char *gradient, unsigned char *cost, int x, int y);
void DisparityCPU(unsigned char* left, unsigned char* right, unsigned char* output, int x, int y, int d);

//
cudaError_t gpuRun(unsigned char* left, unsigned char* right, unsigned char *out, int size, int height, int width);

int getMaxThreads(){
	/*
	CUDA DEVICE prop OUT
	###################################
	*/
	int deviceCount, device;
	int gpuDeviceCount = 0;
	int _maxCudaThreads, _maxCudaProcs, _maxCudaShared, _maxSharedPerBlock;
	struct cudaDeviceProp properties;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
	if (cudaResultCode != cudaSuccess)
		deviceCount = 0;
	/* machines with no GPUs can still report one emulation device */
	for (device = 0; device < deviceCount; ++device) {
		cudaGetDeviceProperties(&properties, device);
		if (properties.major != 9999) /* 9999 means emulation only */
			if (device == 0)
			{
				_maxCudaProcs = properties.multiProcessorCount;
				_maxCudaThreads = properties.maxThreadsPerBlock;
				_maxCudaShared = properties.sharedMemPerMultiprocessor;
				_maxSharedPerBlock = properties.sharedMemPerBlock;
				printf("\n GPU Stats \n # Cuda Processors: %d\n", _maxCudaProcs);
				printf("#max Threads Per Block: %d \n", _maxCudaThreads);
				printf("#Max of shared per block %d\n", _maxSharedPerBlock);
				
			}
	}
	// ----###########################
	return _maxCudaThreads;
};
int getMaxShared(){
	/*
	CUDA DEVICE prop OUT
	###################################
	*/
	int deviceCount, device;
	int gpuDeviceCount = 0;
	int  _maxCudaShared;
	struct cudaDeviceProp properties;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
	if (cudaResultCode != cudaSuccess)
		deviceCount = 0;
	/* machines with no GPUs can still report one emulation device */
	for (device = 0; device < deviceCount; ++device) {
		cudaGetDeviceProperties(&properties, device);
		if (properties.major != 9999) /* 9999 means emulation only */
			if (device == 0)
			{
				_maxCudaShared = properties.sharedMemPerBlock;

			}
	}
	// ----###########################
	return _maxCudaShared;
};
// Main function
int main(){
	char* _filename = "in_left.png";
	char* _filename_2 = "in_right.png";
	char* _out1 = "out_1.bmp";
	char* _out2 = "out_2.bmp";
	char* _out3 = "out_3.bmp";
	char* _out4 = "out_4.bmp";
	int _alpha = 0.9, _rgf = 9, _t1 = 7, _t2 = 2, _dlr = 0, _rmwf = 9, _sigma = 9;
	
	float _epsilon = 255 * 255 * 0.0001;
	float _sigmac = 255 * 0.0001;
	// Basic usage (see HDR discussion below for HDR usage):
	//    int x,y,n;
	//    unsigned char *data = stbi_load(filename, &x, &y, &n, 0);
	//    // ... process data if not NULL ...
	//    // ... x = width, y = height, n = # 8-bit components per pixel ...
	//    // ... replace '0' with '1'..'4' to force that many components per pixel
	//    // ... but 'n' will always be the number that it would have been if you said 0
	//    stbi_image_free(data)
	//
	// Standard parameters:
	//    int *x                 -- outputs image width in pixels
	//    int *y                 -- outputs image height in pixels
	//    int *channels_in_file  -- outputs # of image components in image file
	//    int desired_channels   -- if non-zero, # of image components requested in result
	//
	int x, y, z;
	unsigned char *img_left = stbi_load(_filename, &x, &y, &z, 3);
	unsigned char *img_right = stbi_load(_filename_2, &x, &y, &z, 3);
	//int stbi_write_bmp(char const *filename, int w, int h, int comp, const void *data);
	//unsigned char img_left_b[SIZEX*SIZEY*CHANNEL];
	int size = x*y*z;
	unsigned char *img_left_b = new unsigned char[size];
	unsigned char *img_right_b = new unsigned char[size];
	unsigned char *img_delta = new unsigned char[size];
	unsigned char* img_gradient = new unsigned char[size];
	//
	unsigned char* disparity = new unsigned char[size];

	//Convert to black and white
	ToBlack(img_left, img_left_b,x,y);
	ToBlack(img_right, img_right_b,x,y);
	DisparityCPU(img_left_b, img_right_b, disparity, x, y, DISPARITY_RANGE);
	//
	cudaError_t cudaStatus;
	unsigned char* d_disparity = new unsigned char[size];
	cudaStatus = gpuRun(img_left_b, img_right_b, d_disparity, size, y, x);
	//Delta color
	DeltaColor(img_left, img_right, img_delta,x,y);
	GetGradient(img_left_b, img_right_b, img_gradient,x,y);
	unsigned char *img_cost = new unsigned char[size];
	GetCost(img_delta, img_gradient, img_cost, x, y);
	//
	int write = stbi_write_bmp ( _out1, x, y, 1, img_delta);
	int write2 = stbi_write_bmp( _out2, x, y, 1, img_cost);
	int write3 = stbi_write_bmp( _out3, x, y, 1, disparity);
	int write4 = stbi_write_bmp(_out4, x, y, 1, d_disparity);
	//printf("The output is ? %i \n", write);
	stbi_image_free(img_left);
	stbi_image_free(img_right);
	free(img_delta);
	free(img_left_b);
	free(img_right_b);
	free(img_gradient);
	free(disparity);
	
	return 0;
};

void ToBlack(unsigned char* in, unsigned char* out, int x, int y){
	unsigned char  temp;	
	//Conversion from Color to Black and White
	for (int i = 0; i < x*y; i++){
		temp = 0.299*in[i * 3 + 0] + 0.587*in[i * 3 + 1] + 0.0721*in[i * 3 + 2];
		temp > 255 ? out[i] = 255 : out[i] = temp;
	}
	printf("Calculated Black and White! \n");
}

void DeltaColor(unsigned char* left, unsigned char* right, unsigned char* out, int x, int y){
	unsigned char temp;
	int _t1 = 7;
	for (int i=0; i < x*y; i++){
		temp =(abs(left[i*3+0] - right[i*3+0]) + abs(left[i*3+1] - right[i*3+1]) + abs(left[i*3+2] - right[i*3+2]))/3;
		temp = temp < _t1 ? _t1 : temp;
		temp > 255 ? out[i] = 255 : out[i] = temp;
	}
	printf("Calculated Delta Color! \n");
}

void GetGradient(unsigned char *img_left, unsigned char *img_right, unsigned char *out, int x, int y){

	unsigned char *temp_left, *temp_right;
	temp_left = new unsigned char[x*y*CHANNEL];
	temp_right = new unsigned char[x*y*CHANNEL];

	for (int i = 0; i < x*y; i++){
		if (i==0 || (i+1)%SIZEX==0){
			temp_left[i] = (img_left[i + 1] - img_left[i]) / 2;
			temp_right[i] = (img_right[i + 1] - img_right[i]) / 2;
		}
		else if (i%SIZEX == 0){
			temp_left[i] = (img_left[i] - img_left[i - 1]) / 2;
			temp_right[i] = (img_right[i] - img_right[i - 1]) / 2;
		}
		else{
			temp_left[i] = (img_left[i + 1] - img_left[i - 1]) / 2;
			temp_right[i] = (img_right[i + 1] - img_right[i - 1]) / 2;
		}

		//printf("Gradient is %i : %i\n", temp_left[i], temp_right[i]);
	}
	unsigned char temp;
	char  _t2 = 2;
	for (int i = 0; i < x*y; i++){
		temp = abs(temp_left[i] - temp_right[i]);
		temp < _t2 ? out[i]=_t2 : out[i]=temp;
	}
	printf("Calculated gradient! \n");
}

void GetCost(unsigned char *color, unsigned char *gradient, unsigned char *cost, int x, int y){
	for (int i=0; i < x*y; i++){
		cost[i] = (1 - 0.9)*color[i] + 0.9*gradient[i];
	}
}

void BoxFilter(unsigned char* input,unsigned char* cost, unsigned char* q, int box, int x, int y){
	char r = 9;
	char radius = 2 * r + 1;
	//
	unsigned char *mean, *p, *covar, *pro;
	mean = new unsigned char[x*y];
	p = new unsigned char[x*y];
	covar = new unsigned char[x*y];
	mean = new unsigned char[x*y];
	for (int i = 0; i < x*y; i++){
		// Window
		unsigned char wk=0;
		int sum_m = 0, sum_p=0;

		for (int j = 0; j < radius; j++){
			for (int k = 0; k < radius; k++){
				sum_m += input[i];
				sum_p += cost[i];
				wk++;
			}
		}
		unsigned char mean = sum_m / wk;
	}

}

void DisparityCPU(unsigned char* left, unsigned char* right, unsigned char* output, int x, int y, int d){
	printf("Calculating disparity map for d=%i \n", d);
	for (int i = 0; i < x; i++){
		for (int j = 0; j < y; j++){
			int idxl = i*y+j;
			int idxr = idxl;
			output[idxl] = 255;
			for (int k = -d; k < d; k++){
				int tmp = -1;
				if ((j + k) >= 0 && (j + k) <= x){
				idxr = idxl + k;
				}
				//idxr = idxl + k;
				if (left[idxl] - right[idxr] == 0){
				//	printf("Found disparity range of %i \n", k);
					output[idxl] = k + DISPARITY_RANGE;
				}
			}
		}
	}
	printf("Finished calculation of disparity \n");
}

__global__ void disparityGPU(unsigned char* left, unsigned char* right, unsigned char* out, int height, int width){

	const int shared_size = TILE_DIM*(TILE_DIM+2*DISPARITY_RANGE);
	__shared__ unsigned char s_right[shared_size];

	// Tile start ->(blockIdx.y*gridDim.x + blockIdx.x)*(TILE_DIM*TILE_DIM)
	int tile_idx = (blockIdx.y*gridDim.x + blockIdx.x)*(TILE_DIM*TILE_DIM);
	if(threadIdx.x == 0){
	//Frist thread copies it to shared
		for (int i = 0; i < TILE_DIM; i++){
			for (int j = 0; j <TILE_DIM + 2 * DISPARITY_RANGE; j++){
				int idx_s = i*(TILE_DIM + 2 * DISPARITY_RANGE) + j;
				int idx_r = tile_idx + i*(TILE_DIM + 2 * DISPARITY_RANGE) + j - DISPARITY_RANGE;
				if (idx_r < 0){
					idx_r = 0;
				}
				else if (idx_r > height*width-1){
					idx_r = height*width-1;
				}
				s_right[idx_s] = right[idx_r];

				//printf("\n Thread id %i : %i \n", idx_s, s_right[idx_s]);
			}
		}
	}
	__syncthreads();
	// Disparity calculations&	
	int idx = (blockIdx.y*gridDim.x + blockIdx.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int idx_t = threadIdx.y*TILE_DIM + threadIdx.x;
	unsigned char tmp;
	out[idx] = 255;
	
	for (int i = -DISPARITY_RANGE; i < DISPARITY_RANGE; i++){
		if ((left[idx] - s_right[idx_t + i + DISPARITY_RANGE]) == 0){
			out[idx] = i + DISPARITY_RANGE;
		};
	}
	
}

cudaError_t gpuRun(unsigned char* left, unsigned char* right, unsigned char* out, int size, int height, int width){
	cudaError_t cudaStatus;
	unsigned char *d_left, *d_right, *d_out;
	//allocating memory
	cudaStatus = cudaMalloc((void**)&d_left, size*sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 1 failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_right,  size*sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 2 failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_out, size*sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 3 failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(d_left, left, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaCopy 1 failed! \n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(d_right, right, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaCopy 2 failed! \n");
		goto Error;
	}

	// Memory Data: shared= x^2+14*x-MaxShared;
	int maxShared = getMaxShared();
	int maxTile = 1+(sqrt(DISPARITY_RANGE*DISPARITY_RANGE*4 + 4 * maxShared) - 2 * DISPARITY_RANGE)/2 ;
	int blPerTile = floor(maxTile / BLOCK_SIZE);
	int tileSize = blPerTile*BLOCK_SIZE*(blPerTile*BLOCK_SIZE + DISPARITY_RANGE * 2);
	int tiles =  (height*width) / (TILE_DIM*TILE_DIM);
	printf("\n Running Kernel with %i TileSize %f tiles\n", tileSize, tiles);
	// Threads Data: Th/Block, BlockSize, GridSize
	int max_threads = getMaxThreads();
	dim3 dim_block(TILE_DIM, TILE_DIM); // so your threads are BLOCK_SIZE*BLOCK_SIZE, 256 in this case
	int grid_size = 1+ width*height / (TILE_DIM*TILE_DIM);
	//Running the Kernel
	disparityGPU << < grid_size, dim_block >> >(d_left, d_right, d_out, height, width);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KERNEL failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMemcpy(out, d_out, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaFree(d_right);
	cudaFree(d_left);
	cudaFree(d_out);

	printf(" \n # Successfully run the KERNEL # \n \n");
Error:
	return cudaStatus;
}