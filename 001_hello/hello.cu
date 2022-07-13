#include <stdio.h>


// some helper functions

//extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);


static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// 

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
  // Check the cuda device (
  const int devID = 1;
  cudaDeviceProp props;
  checkCudaErrors(cudaGetDeviceProperties(&props,devID));
  printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           devID, props.name, props.major, props.minor);
  // 
  cuda_hello<<<1,1>>>(); 
  return 0;
}