#include <stdio.h>
#include "../Common/errors.h"


#define N 100


__global__ void add(int *x, int *y, int *z)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  while (id < N) {
    z[id] = x[id] + y[id];
    id += blockDim.x * gridDim.x;
  }
}


int main(void)
{
  int x[N], *device_x;
  HANDLE_ERROR(cudaMalloc((void **) &device_x, sizeof(int) * N));

  int y[N], *device_y;
  HANDLE_ERROR(cudaMalloc((void **) &device_y, sizeof(int) * N));

  int z[N], *device_z;
  HANDLE_ERROR(cudaMalloc((void **) &device_z, sizeof(int) * N));

  for (int i = 0; i < N; i++) {
    x[i] = i;
    y[i] = -i;
  }

  HANDLE_ERROR(cudaMemcpy(device_x, x, sizeof(int) * N,
    cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(device_y, y, sizeof(int) * N,
    cudaMemcpyHostToDevice));

  add<<<128, 128>>>(device_x, device_y, device_z);

  HANDLE_ERROR(cudaMemcpy(z, device_z, sizeof(int) * N,
    cudaMemcpyDeviceToHost));

  int fail_count = 0;  
  for (int i = 0; i < N; i++)
    if (z[i] != x[i] + y[i]) {
      printf("Error: %d + %d != %d\n", x[i], y[i], z[i]);
      fail_count += 1;
    }

  if (fail_count)
    printf("Encountered %d errors\n", fail_count);

  cudaFree(device_x);
  cudaFree(device_y);
  cudaFree(device_z);

  return 0;
}

