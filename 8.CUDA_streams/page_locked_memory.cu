/* Compares the performance of malloc and cudaHostAlloc;
   the latter allocates a buffer of page-blocked host memory
   (also known as pinned memory). This memory is guaranteed
   not to be paged to disk by the OS, thereby making it safe
   for an application to access its physical memory
   (DMA: Direct Memory Access).

   With a single NVIDIA GTX 1070 card on which this program was
   tested, it turns out that the performances are comparable,
   except for device-to-host cudaMemcpy when the host memory
   has been allocated via a call to malloc.
*/


#include <stdio.h>
#include <stdlib.h>
#include "../Common/errors.h"


#define SIZE ((long) 50 * 1024 * 1024) // 50 MB of array allocation


float malloc_test(int, bool);
float cudaHostAlloc_test(int, bool);


int main(void)
{
  float compute_time;
  float size_in_GB = (float) 100 * SIZE * sizeof(int) / 1024 / 1024 / 1024;

  compute_time = malloc_test(SIZE, true);
  printf("GB/s for host-to-device copy using malloc: %.1f\n",
    size_in_GB / (compute_time / 1000));

  compute_time = malloc_test(SIZE, false);
  printf("GB/s for device-to-host copy using malloc: %.1f\n",
    size_in_GB / (compute_time / 1000));

  compute_time = cudaHostAlloc_test(SIZE, true);
  printf("GB/s for host-to-device copy using cudaHostAlloc: %.1f\n",
    size_in_GB / (compute_time / 1000));

  compute_time = cudaHostAlloc_test(SIZE, false);
  printf("GB/s for device-to-host copy using cudaHostAlloc: %.1f\n",
    size_in_GB / (compute_time / 1000));

  return 0;
}


float malloc_test(int n, bool host_to_device)
{
  cudaEvent_t start, stop;
  float compute_time;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  int *array = (int *) malloc(sizeof(int) * n);
  HANDLE_NULL(array);

  int *device_array;
  HANDLE_ERROR(cudaMalloc((void **) &device_array, sizeof(int) * n));

  HANDLE_ERROR(cudaEventRecord(start, 0));
  for (int i = 0; i < 100; i++) {
    if (host_to_device == true)
      HANDLE_ERROR(cudaMemcpy(device_array, array, sizeof(int) * n,
        cudaMemcpyHostToDevice));
    else
      HANDLE_ERROR(cudaMemcpy(array, device_array, sizeof(int) * n,
        cudaMemcpyDeviceToHost));
  }

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&compute_time, start, stop));

  free(array);
  HANDLE_ERROR(cudaFree(device_array));

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  return compute_time;
}


float cudaHostAlloc_test(int n, bool host_to_device)
{
  cudaEvent_t start, stop;
  float compute_time;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  int *array;
  HANDLE_ERROR(cudaHostAlloc((void **) &array, sizeof(int) * n,
    cudaHostAllocDefault));

  int *device_array;
  HANDLE_ERROR(cudaMalloc((void **) &device_array, sizeof(int) * n));

  HANDLE_ERROR(cudaEventRecord(start, 0));
  for (int i = 0; i < 100; i++) {
    if (host_to_device == true)
      HANDLE_ERROR(cudaMemcpy(device_array, array, sizeof(int) * n,
        cudaMemcpyHostToDevice));
    else
      HANDLE_ERROR(cudaMemcpy(array, device_array, sizeof(int) * n,
        cudaMemcpyDeviceToHost));
  }

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&compute_time, start, stop));

  HANDLE_ERROR(cudaFreeHost(array));
  HANDLE_ERROR(cudaFree(device_array));

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  return compute_time;
}
