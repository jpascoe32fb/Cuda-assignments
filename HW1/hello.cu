#include <cuda.h>
#include <iostream>
__global__ void hello() {
    printf("hello");
}

int main() {
    hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}