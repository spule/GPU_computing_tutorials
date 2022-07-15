#include <stdio.h>
#include <string>
#include <iostream>

// some helper functions on error checking

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T> void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// simple kernel

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
  // Check the cuda device -> 0 
  const int devID = 0;
  cudaDeviceProp props;
  checkCudaErrors(cudaGetDeviceProperties(&props,devID));
  printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           devID, props.name, props.major, props.minor);
  // 1. run a simple kernel -> 1 dim grid, 1 dim block
  cuda_hello<<<1,1>>>(); 
  cudaDeviceSynchronize();
  std::cout << std::string(50, '-') << std::endl;
  // 2. same as 1., explicitly define grid and block
  dim3 dgrid(1);
  dim3 dblock(1);
  cuda_hello<<<dgrid,dblock>>>();
  cudaDeviceSynchronize();
  std::cout << std::string(50, '-') << std::endl;
  // 3. increase dimensions of block
  dblock.x = 2; dblock.y = 2; dblock.z = 2;
  cuda_hello << <dgrid, dblock >> > ();
  cudaDeviceSynchronize();
  std::cout << std::string(50, '-') << std::endl;
  // Finish
  return 0;
}