#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "hip_utilities.hpp"

#define HIP_ENABLE_PRINTF

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
}

/*
  block_size -> number of wavefrount = block_size % wavefront_size == 0 ? block_size / wavefront : block_size 
  
  shared memory should be declared by block_size -> communication between threads of one block. Given that all threads of one block will be scheduled on same CU

  wavefront will be scheduled on wavefront pools on this CU. different wavefronts access corresponding part of LDS
*/
__global__
void sumOneDimension (
      float* input,
      float* output,
      size_t size
    )
{
  __shared__ float shared_mem[BLOCK_DIM_X];

  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  shared_mem[threadIdx.x] = input[col];

  __syncthreads();

  for (int stride = BLOCK_DIM_X / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    output[blockIdx.x] = shared_mem[0];
  }
}
