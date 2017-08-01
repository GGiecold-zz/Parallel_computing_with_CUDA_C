/* Introduces the concepts of CUDA streams - which operate like an
   ordered queue for scheduling GPU tasks - and device overlap.
   The latter refers to the possibility of launching a CUDA kernel
   while at the same time a copying between device memory
   and host memory.
   
   If the compute capability of a GPU does not support overlap
   (something we first test within main below), no performance
   improvement could possibly be achieved using CUDA streams.

   We are using cudaHostAlloc to allocate page-blocked host memory,
   as pinned memory is required by cudaMemcpyAsync, a function
   whose use is introduced and illustrated in the code at hand.
   This is called for by the present task-parallelism of executing a
   CUDA C kernel on small chunks of our input arrays (after copying
   those chunks onto the GPU) and copying a corresponding chunk
   of the output array from device to host. Here those chunks
   will be of size BLOCK_SIZE (1MB).
   
   cudaMemcpyAsync differs from memcpy and its CUDA-equivalent
   cudaMemcpy in that the latter two are execute synchronously.
   Similarly, we illustrate an asynchronous kernel launch,
   passing a stream argument to the usual angle-bracketed call.

   The task-parallelism illustrated here is a bit silly but could
   arise in plenty of situations. One that comes to mind would
   involve imbalance between GPU memory and much larger host memory.
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../Common/errors.h"


#define BLOCK_SIZE ((long) 1024 * 1024)
#define SIZE ((long) 50 * BLOCK_SIZE)


__global__ void kernel(int *, int *, int *);


int main(void)
{
  srand(time(NULL));

  cudaEvent_t start, stop;
  float compute_time;
  
  cudaDeviceProp properties;
  int device_id;

  HANDLE_ERROR(cudaGetDevice(&device_id));
  HANDLE_ERROR(cudaGetDeviceProperties(&properties, device_id));

  if (false == (bool) properties.deviceOverlap) {
    printf("The GPU device cannot handle overlaps and therefore no "
           "performance improvement will be achieved via CUDA streams. "
	   "Exiting.\n");

    return 1;
  }

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  cudaStream_t stream;
  HANDLE_ERROR(cudaStreamCreate(&stream));

  int *x, *device_x;
  HANDLE_ERROR(cudaHostAlloc((void **) &x, sizeof(int) * SIZE,
    cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void **) &device_x, sizeof(int) * BLOCK_SIZE));

  int *y, *device_y;
  HANDLE_ERROR(cudaHostAlloc((void **) &y, sizeof(int) * SIZE,
    cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void **) &device_y, sizeof(int) * BLOCK_SIZE));

  int *z, *device_z;
  HANDLE_ERROR(cudaHostAlloc((void **) &z, sizeof(int) * SIZE,
    cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void **) &device_z, sizeof(int) * BLOCK_SIZE));

  for (int i = 0; i < SIZE; i++) {
    x[i] = rand();
    y[i] = rand();
  }

  // Copy the pinned memory from host to device, asynchronously
  // in blocks of size BLOCK_SIZE (1MB as defined at the beginning):
  for (int i = 0; i < SIZE; i += BLOCK_SIZE) {
    HANDLE_ERROR(cudaMemcpyAsync(device_x, x + i, sizeof(int) * BLOCK_SIZE,
      cudaMemcpyHostToDevice, stream));
    HANDLE_ERROR(cudaMemcpyAsync(device_y, y + i, sizeof(int) * BLOCK_SIZE,
      cudaMemcpyHostToDevice, stream));

    kernel<<<BLOCK_SIZE / 256, 256, 0, stream>>>(
      device_x, device_y, device_z);

    HANDLE_ERROR(cudaMemcpyAsync(z + i, device_z, sizeof(int) * BLOCK_SIZE,
      cudaMemcpyDeviceToHost, stream));
  }
  // Makes sure the GPU is done by synchronizing it with the host:
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&compute_time, start, stop));
  printf("Compute time: %.1f ms\n", compute_time);

  HANDLE_ERROR(cudaStreamDestroy(stream));

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  HANDLE_ERROR(cudaFreeHost(x));
  HANDLE_ERROR(cudaFree(device_x));
  HANDLE_ERROR(cudaFreeHost(y));
  HANDLE_ERROR(cudaFree(device_y));
  HANDLE_ERROR(cudaFreeHost(z));
  HANDLE_ERROR(cudaFree(device_z));
  
  return 0;
}


__global__ void kernel(int *x, int *y, int *z)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < SIZE) {
    int id_1 = (id + 1) % 256;
    int id_2 = (id + 2) % 256;

    float sum_x = (x[id] + x[id_1] + x[id_2]) / 3.0f;
    float sum_y = (y[id] + y[id_1] + y[id_2]) / 3.0f;
    
    z[id] = (sum_x + sum_y) / 2.0f;
  }
}
