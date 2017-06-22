#include "includes.h"

using namespace std;

int _alpha = 0.9, _rgf = 9, _t1 = 7, _t2 = 2, _dlr = 0, _rmwf = 9, _sigma = 9;
float _epsilon = 255 * 255 * 0.0001;
float _sigmac = 255 * 0.0001;
void ToBlack(unsigned char* in, unsigned char* out, int x, int y);
void DeltaColor(unsigned char* left, unsigned char* right, unsigned char* out, int x, int y);
void GetGradient(unsigned char *img_left, unsigned char *img_right, unsigned char *out, int x, int y);
void GetCost(unsigned char *color, unsigned char *gradient, unsigned char *cost, int x, int y);
void DisparityCPU(unsigned char* left, unsigned char* right, unsigned char* img_cost, unsigned char* output, int x, int y, int d);

//
cudaError_t gpuRun(unsigned char* left, unsigned char* right, unsigned char *out, int size, int height, int width, unsigned char* left_c, unsigned char* right_c);


//Measure time
double PCFreq;
__int64 CounterStart;

double average(double numbers[], int size);
void StartCounter();
double getCounter();

//
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
	char* _out1 = "delta_color.bmp";
	char* _out3 = "cost_disparity_cpu.bmp";
	char* _out4 = "cost_disparity_gpu.bmp";
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
	unsigned char* disparity = new unsigned char[size];
	unsigned char *img_cost = new unsigned char[size];

	//Convert to black and white
	ToBlack(img_left, img_left_b,x,y);
	ToBlack(img_right, img_right_b,x,y);

	//Reset counter
	PCFreq = 0.0;
	CounterStart = 0;
	clock_t start_t = clock();
	StartCounter();
	//Cpu Algorithm
	DeltaColor(img_left, img_right, img_delta, x, y);
	GetGradient(img_left_b, img_right_b, img_gradient, x, y);
	GetCost(img_delta, img_gradient, img_cost, x, y);
	DisparityCPU(img_left_b, img_right_b, img_cost, disparity, x, y, DISPARITY_RANGE);
	printf("Time taken: %f \n", getCounter());

	//GPU prep, call GPU fct
	cudaError_t cudaStatus;
	unsigned char* d_disparity = new unsigned char[size];
	cudaStatus = gpuRun(img_left_b, img_right_b, d_disparity, size, y, x, img_left, img_right);

	int write = stbi_write_bmp ( _out1, x, y, 1, img_delta);
	int write3 = stbi_write_bmp( _out3, x, y, 1, disparity);
	int write4 = stbi_write_bmp(_out4, x, y, 1, d_disparity);

	//CPU Free
	stbi_image_free(img_left);
	stbi_image_free(img_right);
	free(img_delta);
	free(img_left_b);
	free(img_right_b);
	free(img_gradient);
	free(disparity);
	free(img_cost);
	free(d_disparity);
	
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
	for (int i=0; i < x*y; i++){
		temp =(abs(left[i*3+0] - right[i*3+0]) + abs(left[i*3+1] - right[i*3+1]) + abs(left[i*3+2] - right[i*3+2]))/3;
		temp = temp < _t1 ? _t1 : temp;
		temp > 255 ? out[i] = 255 : out[i] = temp;
	}
	printf("Calculated Delta Color! \n");
}

void GetGradient(unsigned char *img_left, unsigned char *img_right, unsigned char *out, int x, int y){

	float l_grad, r_grad;

	for (int i = 0; i < x*y; i++){
		if (i==0 || (i+1)%SIZEX==0){
			l_grad = (img_left[i + 1] - img_left[i]) / 2;
			r_grad = (img_right[i + 1] - img_right[i]) / 2;
		}
		else if (i%SIZEX == 0){
			l_grad = (img_left[i] - img_left[i - 1]) / 2;
			r_grad = (img_right[i] - img_right[i - 1]) / 2;
		}
		else{
			l_grad = (img_left[i + 1] - img_left[i - 1]) / 2;
			r_grad = (img_right[i + 1] - img_right[i - 1]) / 2;
		}
		int grad_n = (int)abs(l_grad - r_grad);
		out[i] = MIN(grad_n, _t2);
	}
	printf("Calculated gradient! \n");
}

void GetCost(unsigned char *color, unsigned char *gradient, unsigned char *cost, int x, int y){
	for (int i=0; i < x*y; i++){
		cost[i] = (1 - _alpha)*color[i] + _alpha*gradient[i];
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
	unsigned char wk = 0;
	int sum_m = 0, sum_p = 0;

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

void DisparityCPU(unsigned char* left, unsigned char* right, unsigned char* img_cost, unsigned char* output, int x, int y, int d){
	printf("Calculating disparity map for d=%i \n", d);
	for (int i = 0; i < x; i++){
		for (int j = 0; j < y; j++){
			int idxl = i*y + j;
			int idxr = idxl;
			output[idxl] = (1-_alpha)*_t1 + _alpha*_t2;
			for (int k = -d; k < d; k++){
				int tmp = -1;
				if ((j + k) >= 0 && (j + k) <= x){
					idxr = idxl + k;
				}
				if (left[idxl] - right[idxr] == 0){
					output[idxl] = img_cost[idxr];
				}
			}
		}
	}
	printf("Finished calculation of disparity \n");
}

__global__ void disparityGPU(unsigned char* left, unsigned char* right, unsigned char* out, int height, int width, unsigned char* grad, unsigned char* color){
	int _alpha = 0.9, _t1 = 7, _t2 = 2;
	const int shared_size = TILE_DIM*(TILE_DIM + 2 * DISPARITY_RANGE);
	__shared__ unsigned char s_right[shared_size];

	// Tile start ->(blockIdx.y*gridDim.x + blockIdx.x)*(TILE_DIM*TILE_DIM)
	int tile_idx = (blockIdx.y*gridDim.x + blockIdx.x)*(TILE_DIM*TILE_DIM);
	//From main to shared
	for (int j = 0; j < TILE_DIM + 2 * DISPARITY_RANGE; j++){
		int idx_s = threadIdx.x*(TILE_DIM + 2 * DISPARITY_RANGE) + j;
		int idx_r = tile_idx + threadIdx.x*(TILE_DIM + 2 * DISPARITY_RANGE) + j - DISPARITY_RANGE;
		if (idx_r < 0){
			idx_r = 0;
		}
		else if (idx_r > height*width - 1){
			idx_r = height*width - 1;
		}
		s_right[idx_s] = right[idx_r];

	}
	__syncthreads();
	// Disparity calculations&	
	int idx = (blockIdx.y*gridDim.x + blockIdx.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int idx_t = threadIdx.y*TILE_DIM + threadIdx.x;
	out[idx] = (1-_alpha)*_t1+_alpha*_t2;
	for (int i = -DISPARITY_RANGE; i < DISPARITY_RANGE; i++){
		if ((left[idx] - s_right[idx_t + i + DISPARITY_RANGE]) == 0){
			out[idx] = (1 - _alpha)*color[idx + i] + _alpha*grad[idx + i];
		};
	}

}

__global__ void colorDeltaGPU(unsigned char* left, unsigned char* right, unsigned char* delta){
	int _t1 = 7;
	int idx = (blockIdx.y*gridDim.x + blockIdx.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	unsigned char temp = (abs(left[idx * 3 + 0] - right[idx * 3 + 0]) + abs(left[idx * 3 + 1] - right[idx * 3 + 1]) + abs(left[idx * 3 + 2] - right[idx * 3 + 2])) / 3;
	temp = temp < _t1 ? _t1 : temp;
	temp > 255 ? delta[idx] = 255 : delta[idx] = temp;
}

__global__ void gradGPU(unsigned char* left, unsigned char* right, unsigned char* grad, int height, int width){
	int _t2 = 2;
	int idx = (blockIdx.y*gridDim.x + blockIdx.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int grad_n;
	float grad_r, grad_l;
	if (idx == 0 || idx%(width+1)==0){
		grad_l = (left[idx + 1] - left[idx]) / 2;
		grad_r = (right[idx + 1] - left[idx]) / 2;
	}
	else if (idx == width*height || idx % width == 0){
		grad_l = (left[idx] - left[idx - 1]) / 2;
		grad_r = (right[idx] - left[idx - 1]) / 2;
	}
	else{
		grad_l = (left[idx + 1] - left[idx - 1]) / 2;
		grad_r = (right[idx + 1] - left[idx - 1]) / 2;
	}

	grad_n = (int)abs(grad_l - grad_r);
	grad[idx] = (grad_n < _t2) ? grad_n : _t2;
	
}

cudaError_t gpuRun(unsigned char* left, unsigned char* right, unsigned char* out, int size, int height, int width, unsigned char* left_c, unsigned char* right_c){

	cudaError_t cudaStatus;
	unsigned char *d_left, *d_right, *d_out, *d_color, *d_grad, *d_cleft, *d_cright;

	// Threads Data: Th/Block, BlockSize, GridSize
	int max_threads = getMaxThreads();
	dim3 dim_block(TILE_DIM, TILE_DIM); // so your threads are BLOCK_SIZE*BLOCK_SIZE, 256 in this case
	int grid_size = 1 + width*height / (TILE_DIM*TILE_DIM);

	//Chorno
	PCFreq = 0.0;
	CounterStart = 0;
	clock_t start_t = clock();
	// Cuda alocation
	cudaMalloc((void**)&d_cleft, size*sizeof(unsigned char));
	cudaMalloc((void**)&d_cright, size*sizeof(unsigned char));
	cudaMemcpy(d_cleft, left_c, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cright, right_c, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaMalloc((void**)&d_color, size*sizeof(unsigned char)) != cudaSuccess){
		goto Error;
	}

	StartCounter();
	colorDeltaGPU << <grid_size, dim_block >> > (d_cleft, d_cright, d_color);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KERNEL Color failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaFree(d_cleft);
	cudaFree(d_cright);

	cudaMalloc((void**)&d_grad, size*sizeof(unsigned char));
	//allocating memory
	cudaMalloc((void**)&d_left, size*sizeof(unsigned char));
	cudaMalloc((void**)&d_right, size*sizeof(unsigned char));
	cudaMalloc((void**)&d_out, size*sizeof(unsigned char));

	// Memory Data: shared= x^2+14*x-MaxShared; calcules inutiles
	int maxShared = getMaxShared();
	int maxTile = 1+(sqrt(DISPARITY_RANGE*DISPARITY_RANGE*4 + 4 * maxShared) - 2 * DISPARITY_RANGE)/2 ;
	int blPerTile = floor(maxTile / BLOCK_SIZE);
	int tileSize = blPerTile*BLOCK_SIZE*(blPerTile*BLOCK_SIZE + DISPARITY_RANGE * 2);
	int tiles =  (height*width) / (TILE_DIM*TILE_DIM);


	//printf("Time memory allocation&& copy HostToDevice: %f \n", getCounter());
	//StartCounter();
	//printf("Blocks %i\n \n", grid_size);
	//Memory COpying
	if (cudaMemcpy(d_left, left, size*sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess){
			printf("Failed mem copy \n");
			goto Error;
		}
	if (cudaMemcpy(d_right, right, size*sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess){
			printf("Failed mem copy rightr \n");
			goto Error;
		}
		gradGPU << < grid_size, dim_block >> >(d_left, d_right, d_grad, height, width);
		cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "KERNEL Grad failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
	//Kernel Run
		disparityGPU << < grid_size, dim_block >> >(d_left, d_right, d_out, height, width, d_grad, d_color);
		cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "KERNEL failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
	//cudaDeviceSynchronize();
	//printf("Time kernel run: %f \n", getCounter());
	//StartCounter();
		cudaStatus = cudaMemcpy(out, d_out, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess){
		printf("Failed to copy");
		goto Error;
	}
	printf("Total Tiem for gpu run : memcopies + kernel: %f \n", getCounter());
	cudaFree(d_right);
	cudaFree(d_left);
	cudaFree(d_out);

Error:
	return cudaStatus;
}


void StartCounter(){
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		std::cout << "QueryPerformanceFrequency failed! \n";

	PCFreq = double(li.QuadPart);// / 1000000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}
double getCounter(){
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - CounterStart) / PCFreq;
}