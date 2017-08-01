/* Histogram computation from an array of random bytes, using
   GPU hardware-supported global memory atomics for avoiding
   thread read and write access conflicts in the updates to
   our histogram's 256 bins.
   
   Namely, the code herewith mostly introduces the atomic sequence of operations
   encompassed by CUDA C's atomicAdd(address, value).
   
   However, the performance is poor compared to a simple CPU implementation;
   this is improved by using shared instead of global memory, as illustrated
   in histogram_with_shared_memory_atomics.cu, part of the same directory.
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../Common/errors.h"


#define BINS 256
#define SIZE ((long int) (100 * 1024 * 1024)) // 100MB of random bytes


__global__ void kernel(unsigned char *, long, unsigned int *);
unsigned char* random_bytes_stream(int);


int main(void)
{
  cudaEvent_t start, stop;
  float compute_time;
  
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  srand(time(NULL));
  unsigned char *stream = random_bytes_stream(SIZE);

  unsigned char *device_stream;
  unsigned int *device_histogram;

  HANDLE_ERROR(cudaMalloc((void **) &device_stream, SIZE));
  HANDLE_ERROR(cudaMemcpy(device_stream, stream, SIZE,
    cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void **) &device_histogram,
    sizeof(unsigned int) * BINS));
  HANDLE_ERROR(cudaMemset(device_histogram, 0,
    sizeof(unsigned int) * BINS));

  cudaDeviceProp properties;
  HANDLE_ERROR(cudaGetDeviceProperties(&properties, 0));
  kernel<<<4 * properties.multiProcessorCount, BINS>>>(
    device_stream, SIZE, device_histogram);

  unsigned int *histogram = new unsigned int[BINS]();
  HANDLE_ERROR(cudaMemcpy(histogram, device_histogram, sizeof(int) * BINS,
    cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&compute_time, start, stop));
  printf("Histogram computation took %.1f ms\n", compute_time);

  long total = 0;
  for (int i = 0; i < BINS; i++)
    total += histogram[i];
    
  printf("Total sum of histogram bins vs. expected: %ld vs. %ld",
    total, SIZE);

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  
  HANDLE_ERROR(cudaFree(device_stream));
  HANDLE_ERROR(cudaFree(device_histogram));
  delete [] histogram;

  return 0;
}


__global__ void kernel(
  unsigned char *stream, long size, unsigned int *histogram)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  while (id < size) {
    atomicAdd(&(histogram[stream[id]]), 1);
    id += blockDim.x * gridDim.x;
  }
}


unsigned char* random_bytes_stream(int n)
{
  unsigned char *stream = (unsigned char *) malloc(n);
  HANDLE_NULL(stream);

  for (int i = 0; i < n; i++)
    stream[i] = rand();

  return stream;
}
