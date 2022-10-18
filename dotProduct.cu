#include <stdio.h>
//#include <sys/time.h>

const int threads_per_block = 512;

// Forward function declarations
float GPU_big_dot(float *A, float *B, int N, float *time);
float GPU_big_dot2(float *A, float *B, int N, float *time);
float *get_random_vector(int N);
void die(const char *message);

int main(int argc, char **argv) {
	// Seed the random generator (use a constant here for repeatable results)
	srand(10);

	// Determine the vector length
	int N = 1000;  // default value
	//int N = 1 << 24;  // default value
	if (argc > 1) N = atoi(argv[1]); // user-specified value

	// Generate two random vectors
	float *A = get_random_vector(N);
	float *B = get_random_vector(N);
	
	// Compute their dot product using GPU1 kernal
	float GPU1_Time[1] = {0};
	float GPU1 = GPU_big_dot(A, B, N, GPU1_Time);
	
	// Compute their dot product using GPU2 kernal
	float GPU2_Time[1] = {0};
	float GPU2 = GPU_big_dot2(A, B, N, GPU2_Time);
	
	// Compute the speedup or slowdown
	if (GPU1_Time[0] > GPU2_Time[0]) printf("\nThe kernal with atomics outperformed the kernal without it by %.2fx\n", GPU1_Time[0] / GPU2_Time[0]);
	else                     printf("\nThe kernal without atomics outperformed the kernal with it by %.2fx\n", GPU2_Time[0] / GPU1_Time[0]);
	
	// Check the correctness of the GPU results
        if (fabs(GPU1 - GPU2) > 0.000001) 
	  printf("\nvalues incorrect, CPU dot product = %f, GPU dot product = %f\n", GPU1, GPU2);
	else           
	  printf("\nvalues correct, CPU dot product = %f, GPU dot product = %f\n", GPU1, GPU2);

}

// A GPU kernel that computes the vector dot product of A and B
// (uses shared mem and parallel reduction)
__global__ void dot_product_kernel1(float *a, float *b, float *out, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
    int block_diff = blockDim.x*gridDim.x;

    __shared__ float cache[threads_per_block];

    //float temp = 0.0f; /////this is the issue why its off/////////////////////
    while(index < n) {
        cache[threadIdx.x] += a[index] * b[index] ;
        index += block_diff;
    }

    //cache[threadIdx.x] = temp;

    __syncthreads();

    unsigned int i = blockDim.x/2;
    while(i != 0) {
        if(threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

	if(threadIdx.x == 0) out[blockIdx.x] = cache[0];
}

// Returns the vector dot product of A and B
// Calls kernal1
float GPU_big_dot(float *A_CPU, float *B_CPU, int N, float *time) {	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// Allocate GPU memory for the inputs and the result
	int vector_size = N * sizeof(float);
	float *A_GPU, *B_GPU, *GPU2;
	if (cudaMalloc((void **) &A_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	if (cudaMalloc((void **) &B_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	if (cudaMalloc((void **) &GPU2, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	
	// Transfer the input vectors to GPU memory
	cudaMemcpy(A_GPU, A_CPU, vector_size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, B_CPU, vector_size, cudaMemcpyHostToDevice);
		
	// Determine the number of thread blocks in the grid 
	int blocks_per_grid = (int) ((float) (N + threads_per_block - 1) / (float) threads_per_block);
	
	// Execute the kernel to compute the vector dot product on the GPU
	cudaEventRecord(start);
	dot_product_kernel1<<< blocks_per_grid , threads_per_block >>> (A_GPU, B_GPU, GPU2, N);
	cudaDeviceSynchronize(); 
	cudaEventRecord(stop);
	
	
	// Check for kernel errors
	cudaError_t error = cudaGetLastError();
	if (error) {
	  char message[256];
	  sprintf(message, "CUDA error: %s", cudaGetErrorString(error));
	  die(message);
	}
	
	// Allocate CPU memory for the result
	float *GPU1 = (float *) malloc(vector_size);
	if (GPU1 == NULL) die("Error allocating CPU memory");
	
	// Transfer the result from the GPU to the CPU
	cudaMemcpy(GPU1, GPU2, vector_size, cudaMemcpyDeviceToHost);

	// Get time
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	time[0] = milliseconds;
	
	// Free the GPU memory
	cudaFree(A_GPU);
	cudaFree(B_GPU);
	cudaFree(GPU2);
	
	//Do summation of multiplication in CPU
	float sum = 0;
	for(int i = 0; i < N; i++) sum += GPU1[i];

	return sum;
}

// A GPU kernel that computes the vector dot product of A and B
// (uses shared mem, parallel reduction, and atomics)
__global__ void dot_product_kernel2(float *a, float *b, float *out, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
    int block_diff = blockDim.x*gridDim.x;

    __shared__ float cache[threads_per_block];

    float temp = 0.0f; /////this is the issue why its off/////////////////////
    while(index < n) {
        temp += a[index] * b[index] ;
        index += block_diff;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();

    unsigned int i = blockDim.x/2;
    while(i != 0) {
        if(threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    
    if(threadIdx.x == 0) {
        atomicAdd(out, cache[0]);
    }
}

// Returns the vector dot product of A and B (computed on the GPU)
// Calls kernal2
float GPU_big_dot2(float *A_CPU, float *B_CPU, int N, float time[]) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// Allocate GPU memory for the inputs and the result
	int vector_size = N * sizeof(float);
	float *A_GPU, *B_GPU, *GPU2;
	if (cudaMalloc((void **) &A_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	if (cudaMalloc((void **) &B_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	if (cudaMalloc((void **) &GPU2, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	
	// Transfer the input vectors to GPU memory
	cudaMemcpy(A_GPU, A_CPU, vector_size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, B_CPU, vector_size, cudaMemcpyHostToDevice);
		
	// Determine the number of thread blocks in the grid 
	int blocks_per_grid = (int) ((float) (N + threads_per_block - 1) / (float) threads_per_block);
	
	// Execute the kernel to compute the vector dot product on the GPU
	cudaEventRecord(start);
	dot_product_kernel2<<< blocks_per_grid , threads_per_block >>> (A_GPU, B_GPU, GPU2, N);
	cudaDeviceSynchronize(); 
	cudaEventRecord(stop);
	
	// Check for kernel errors
	cudaError_t error = cudaGetLastError();
	if (error) {
	  char message[256];
	  sprintf(message, "CUDA error: %s", cudaGetErrorString(error));
	  die(message);
	}
	
	// Allocate CPU memory for the result
	float *GPU1 = (float *) malloc(vector_size);
	if (GPU1 == NULL) die("Error allocating CPU memory");
	
	// Transfer the result from the GPU to the CPU
	cudaMemcpy(GPU1, GPU2, vector_size, cudaMemcpyDeviceToHost);

	// Get time
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	time[0] = milliseconds;
	
	// Free the GPU memory
	cudaFree(A_GPU);
	cudaFree(B_GPU);
	cudaFree(GPU2);
	
	//return sum from kernal
	return GPU1[0];
}

// Returns a randomized vector containing N elements
float *get_random_vector(int N) {
	if (N < 1) die("Number of elements must be greater than zero");
	
	// Allocate memory for the vector
	float *V = (float *) malloc(N * sizeof(float));
	if (V == NULL) die("Error allocating CPU memory");
	
	// Populate the vector with random numbers
	for (int i = 0; i < N; i++) V[i] = (float) rand() / (float) rand();
	
	// Return the randomized vector
	return V;
}

// Prints the specified message and quits
void die(const char *message) {
	printf("%s\n", message);
	exit(1);
}