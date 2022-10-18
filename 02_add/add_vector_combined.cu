#include <stdio.h>
#include <stdlib.h>

#define N 1024*1024 
#define THREADS_PER_BLOCK 512

__global__ void add(int *a, int *b, int *c, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n)
    c[index] = a[index] + b[index];
}

void random_ints(int *x, int size)
{
  for (int i = 0; i < size; i++) {
    x[i] = rand() % 10;
  }
}

int main(void) {
  int *a, *b, *c; // host copies of a, b, c
  int *d_a, *d_b, *d_c; // device copies of a, b, c
  int size = N * sizeof(int);

  // Allocate space for device copies of a, b, c
  cudaMalloc((void **) &d_a, size);
  cudaMalloc((void **) &d_b, size);
  cudaMalloc((void **) &d_c, size);

  // Alloc space for host copies of a, b, c and setup input values
  a = (int *) malloc(size);
  random_ints(a, N);
  b = (int *) malloc(size);
  random_ints(b, N);
  c = (int *) malloc(size);
  
  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Launch add() kernel on GPU with N threads 
  add<<<(N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
  //add<<<(N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, 102500>>>(d_a, d_b, d_c, N);

  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) 
    printf("a[%d] = %d,  b[%d] = %d, c[%d] = %d\n", i, a[i], i, b[i], i, c[i]);

  // Cleanup
  cudaFree(d_a); 
  cudaFree(d_b);
  cudaFree(d_c);
  
  return 0;
}

