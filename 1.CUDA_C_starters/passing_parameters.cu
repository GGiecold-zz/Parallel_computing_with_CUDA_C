#include <stdio.h>
#include "../Common/errors.h"


__global__ void add(int i, int j, int *k)
{
  *k = i + j;
}

int main(void)
{
  int i;
  int *device_ptr;

  HANDLE_ERROR(cudaMalloc((void **) &device_ptr, sizeof(int)));
  add<<<1, 1>>>(2, 7, device_ptr);
  HANDLE_ERROR(cudaMemcpy(&i, device_ptr, sizeof(int), cudaMemcpyDeviceToHost));
  
  printf("2 + 7 = %d\n", i);

  cudaFree(device_ptr);

  return 0;
}
