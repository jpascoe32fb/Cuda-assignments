#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <sys/time.h>


#define N 100024
//#define err 1.0e-6

/*long long start_timer() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

long long stop_timer(long long start_time, const char *name) {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;

    printf("%s: %.5f sec\n", name, ((float) (end_time - start_time)) / (1000 * 1000));

    return end_time - start_time;
}*/

__host__ float CPU_big_dot(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++) {
        out[0] += a[i] + b[i];
    }
    
    return *out;
}

__device__ float GPU_big_dot(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++) {
        out[0] += a[i] + b[i];
    }
    
    return *out;
}

__global__ void Device(float *out, float *a, float *b, int n) {
    //long long start_time = start_timer();
    *out = GPU_big_dot(out, a, b, n);
    //long long end_time = stop_timer(start_time, "GPU");
}

int main() {
    float *a,*b,*cpu_out,*gpu_out;
    float *d_a, *d_b, *d_out;

    //Create memory space for host vars
    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    cpu_out = (float*)malloc(sizeof(float) * N);
    gpu_out = (float*)malloc(sizeof(float) * N);

    //fill host var vectors with values 
    for(int i = 0;i < N; i++) {
        a[i] = 1.3f; b[i] = 2.8f;
    }

    //allocate memory on device for vars
    cudaMalloc((void**)&d_a, sizeof(float) *N);
    cudaMalloc((void**)&d_b, sizeof(float) *N);
    cudaMalloc((void**)&d_out, sizeof(float) *N);

    //copy vals of host vars to device
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    //call gpu
    Device<<<1,1>>>(d_out,d_a,d_b,N);
    cudaMemcpy(gpu_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    printf("GPU sum := %.8g\n", gpu_out[0]);
    cudaDeviceSynchronize();

    CPU_big_dot(cpu_out,a,b,N);
    printf("CPU sum := %.8g\n", cpu_out[0]);


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(a);
    free(b);
    free(cpu_out);
    free(gpu_out);
}