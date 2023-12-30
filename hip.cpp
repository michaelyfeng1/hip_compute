#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "hip_utilities.hpp"

#define HIP_ENABLE_PRINTF


__global__ 
void reductionSum (
      float* input,
      float* input2,
      float* output,
      size_t n 
    )
{
  float data;
  unsigned int i = blockIdx.x * blockDim.x +  threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  output[j * n + i] = input[j * n + i] + input2[j * n + i];;
}

__global__
void matrixMul (
      float* input0,
      float* input1,
      float* output,
      size_t n, //assume square matrix
      size_t m
    )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;//col
  int j = blockIdx.y * blockDim.y + threadIdx.y;//row 
  
  float sum = 0;

  for (int index = 0; index < m; index++) {
    sum += input0[j * n + index] * input1[index * n + j];
  }

  output[j*m + i] = sum;

  __threadfence();
}

__global__
void matrixUpdate (
      float* input0,
      float* input1,
      float* output,
      size_t m,
      size_t n
    )
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  int index = i * m  + j;
 
  output[index] = input0[index] + input1[index];
  
  __threadfence();
  //__syncthreads();
}
