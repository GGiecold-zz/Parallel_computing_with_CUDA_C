#include <stdio.h>
#include "../Common/errors.h"

/* Assuming that our application requires double-precision
   floating-point support, we need to identify at least one card
   with compute capability 1.3 or higher.

   The short code herewith shows a more convenient way of doing so
   than iterating over all devices via cudaGetDeviceCount() and
   accessing their properties via cudaGetDeviceProperties().
*/

int main(void)
{
  int device_id;
  
  HANDLE_ERROR(cudaGetDevice(&device_id));
  printf("ID of current CUDA device: %d\n", device_id);

  cudaDeviceProp properties;
  memset(&properties, 0, sizeof(cudaDeviceProp));
  properties.major = 1;
  properties.minor = 3;
  
  HANDLE_ERROR(cudaChooseDevice(&device_id, &properties));
  printf("ID of CUDA device closest to our capability requirements: %d\n",
    device_id);

  HANDLE_ERROR(cudaSetDevice(device_id));
  
  return 0;
}
