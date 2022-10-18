#include "stdio.h"

__global__ void mykernel(void) {
    printf( "Hello, World, I'm GPU thread [%d, %d]!\n", blockIdx.x, threadIdx.x);
}

int main( void ) {
    mykernel<<<2,3>>>();
    cudaDeviceSynchronize();
    printf( "Hello, World, I'm CPU thread!\n" );
    return 0;
}

