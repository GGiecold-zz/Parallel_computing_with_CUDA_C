/* Illustrates task-parallelism and asynchronous kernel launch and
   memory copies using two CUDA streams, i.e. two independent queues
   of operations scheduled for execution on the GPU.
   Improved performance is achieved using the overlap capacity offered
   by most current GPU devices of doing kernel execution with memory
   copies between device and host in parallel. This is achieved through
   the GPU hardware having separate engines for those two executions.

   For more information on the mock chunking performed by the code
   herewith, see task_parallelism_with_single_CUDA_stream.cu,
   part of the same directory.
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

  cudaStream_t stream_0, stream_1;
  HANDLE_ERROR(cudaStreamCreate(&stream_0));
  HANDLE_ERROR(cudaStreamCreate(&stream_1));

  // We allocate memory for host arrays and corresponding GPU buffers
  // associated with stream_0 and stream_1:
  
  int *x, *device_x_0, *device_x_1;
  HANDLE_ERROR(cudaHostAlloc((void **) &x, sizeof(int) * SIZE,
    cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void **) &device_x_0,
    sizeof(int) * BLOCK_SIZE));
  HANDLE_ERROR(cudaMalloc((void **) &device_x_1,
    sizeof(int) * BLOCK_SIZE));

  int *y, *device_y_0, *device_y_1;
  HANDLE_ERROR(cudaHostAlloc((void **) &y, sizeof(int) * SIZE,
    cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void **) &device_y_0,
    sizeof(int) * BLOCK_SIZE));
  HANDLE_ERROR(cudaMalloc((void **) &device_y_1,
    sizeof(int) * BLOCK_SIZE));

  int *z, *device_z_0, *device_z_1;
  HANDLE_ERROR(cudaHostAlloc((void **) &z, sizeof(int) * SIZE,
    cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void **) &device_z_0,
    sizeof(int) * BLOCK_SIZE));
  HANDLE_ERROR(cudaMalloc((void **) &device_z_1,
    sizeof(int) * BLOCK_SIZE));

  for (int i = 0; i < SIZE; i++) {
    x[i] = rand();
    y[i] = rand();
  }

  // Naively, we would copy chunks of the pinned host memory to device,
  // launch the kernel, then copy the output array in chunk
  // from device to host.
  // Those steps would be performed asynchronously by alternating with
  // two streams.
  // However, scheduling all of a given stream's operations at once
  // is often inefficient, with some copies and kernel
  // execution being blocked by others and waiting idly.
  // To bypass the performance bottleneck that would result,
  // it turns out to be more efficient to interleave
  // the copy and kernel execution tasks across streams,
  // as is done in the following lines of code:
  
  for (int i = 0; i < SIZE; i += 2 * BLOCK_SIZE) {
    HANDLE_ERROR(cudaMemcpyAsync(device_x_0, x + i, sizeof(int) * BLOCK_SIZE,
      cudaMemcpyHostToDevice, stream_0));
    HANDLE_ERROR(cudaMemcpyAsync(device_x_1, x + i + BLOCK_SIZE,
      sizeof(int) * BLOCK_SIZE, cudaMemcpyHostToDevice, stream_1));
      
    HANDLE_ERROR(cudaMemcpyAsync(device_y_0, y + i, sizeof(int) * BLOCK_SIZE,
      cudaMemcpyHostToDevice, stream_0));
    HANDLE_ERROR(cudaMemcpyAsync(device_y_1, y + i + BLOCK_SIZE,
      sizeof(int) * BLOCK_SIZE, cudaMemcpyHostToDevice, stream_1));

    kernel<<<BLOCK_SIZE / 256, 256, 0, stream_0>>>(
      device_x_0, device_y_0, device_z_0);
    kernel<<<BLOCK_SIZE / 256, 256, 0, stream_1>>>(
      device_x_1, device_y_1, device_z_1);

    HANDLE_ERROR(cudaMemcpyAsync(z + i, device_z_0, sizeof(int) * BLOCK_SIZE,
      cudaMemcpyDeviceToHost, stream_0));
    HANDLE_ERROR(cudaMemcpyAsync(z + i + BLOCK_SIZE, device_z_1,
      sizeof(int) * BLOCK_SIZE, cudaMemcpyDeviceToHost, stream_1));
  }
  // Makes sure the GPU streams are done by synchronizing with the CPU:
  HANDLE_ERROR(cudaStreamSynchronize(stream_0));
  HANDLE_ERROR(cudaStreamSynchronize(stream_1));

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&compute_time, start, stop));
  printf("Compute time: %.1f ms\n", compute_time);

  // Lots of cleanup:
  
  HANDLE_ERROR(cudaStreamDestroy(stream_0));
  HANDLE_ERROR(cudaStreamDestroy(stream_1));

  HANDLE_ERROR(cudaFreeHost(x));
  HANDLE_ERROR(cudaFree(device_x_0));
  HANDLE_ERROR(cudaFree(device_x_1));
  
  HANDLE_ERROR(cudaFreeHost(y));
  HANDLE_ERROR(cudaFree(device_y_0));
  HANDLE_ERROR(cudaFree(device_y_1));
  
  HANDLE_ERROR(cudaFreeHost(z));
  HANDLE_ERROR(cudaFree(device_z_0));
  HANDLE_ERROR(cudaFree(device_z_1));

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

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
