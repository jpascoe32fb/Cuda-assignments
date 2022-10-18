#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void increment(int *d_x) {
  *d_x += 1;
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

