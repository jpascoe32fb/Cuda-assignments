#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void increment(int *d_x) {
  atomicAdd(d_x, 1); 
}

int main(){
  int x = 0, *d_x;

  cudaMalloc((void**) &d_x, sizeof(int));
  cudaMemcpy(d_x, &x, sizeof(int), cudaMemcpyHostToDevice);

  increment<<<1000,1000>>>(d_x);

  cudaMemcpy(&x, d_x, sizeof(int), cudaMemcpyDeviceToHost);

  printf("x = %d\n", x);
  cudaFree(d_x);
}

[13:05:30] jin6@titan1:~/CUDA/RaceCondition [69] nvcc  -Wno-deprecated-gpu-targets raceConditionFixed.cu
[13:06:28] jin6@titan1:~/CUDA/RaceCondition [70] ./a.out
x = 1000000



