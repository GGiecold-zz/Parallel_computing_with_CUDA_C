#include <stdio.h>
#include "../Common/errors.h"

/* cudaGetDeviceCount() allows us to determine the number of available
 devices supporting a CUDA architecture.
 The properties of those devices are stored in a struct cudaDeviceProp,
 as returned by a call to cudaGetDeviceProperties().
 Information on the public variables of this data structure can be found
 at https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
*/

int main(void)
{
  int i, count_devices;
  HANDLE_ERROR(cudaGetDeviceCount(&count_devices));

  cudaDeviceProp properties;
  for (i = 0; i < count_devices; i++) {
    HANDLE_ERROR(cudaGetDeviceProperties(&properties, i));
    
    printf("Properties of device %d\n", i);
    printf("Name: %s\n", properties.name);
    printf("Clock rate: %d\n", properties.clockRate);
    printf("Major and minor compute capabilities: %d, %d\n",
      properties.major, properties.minor);
      
    printf("Can the device concurrently copy memory and execute a kernel? ");
    if (properties.asyncEngineCount)
      printf("Yes\n");
    else
      printf("No, disabled\n");

    printf("Kernel execution timeout: ");
    if (properties.kernelExecTimeoutEnabled)
      printf("enabled\n");
    else
      printf("disabled\n");

    printf("Total global memory available (bytes): %ld\n",
      properties.totalGlobalMem);
    printf("Total constant available memory (bytes): %ld\n",
      properties.totalConstMem);

    printf("Number of multiprocessors on device: %d\n",
      properties.multiProcessorCount);
    printf("Shared memory per multiprocessor: %d\n",
      properties.sharedMemPerBlock);
    printf("Maximum number of threads per block: %d\n",
      properties.maxThreadsPerBlock);
    printf("Maximum size of each dimension of a block: (%d, %d, %d)\n",
      properties.maxThreadsDim[0],
      properties.maxThreadsDim[1],
      properties.maxThreadsDim[2]);

    printf("\n");
  }

  return 0;
}
