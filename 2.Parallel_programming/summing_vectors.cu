#include <stdio.h>
#include "../Common/errors.h"


#define N 100


__global__ void add(int *v1, int *v2, int *v3)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (id < N)
    v3[id] = v1[id] + v2[id];
}


int main(void)
{
  int v1[N];
  int *device_v1;
  HANDLE_ERROR(cudaMalloc((void **) &device_v1, sizeof(int) * N));

  int v2[N];
  int *device_v2;
  HANDLE_ERROR(cudaMalloc((void **) &device_v2, sizeof(int) * N));

  int v3[N];
  int *device_v3;
  HANDLE_ERROR(cudaMalloc((void **) &device_v3, sizeof(int) * N));

  for (int i = 0, j = N - 1; i < N; i++, j--) {
    v1[i] = i;
    v2[i] = j;
  }

  HANDLE_ERROR(cudaMemcpy(device_v1, v1, sizeof(int) * N,
    cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(device_v2, v2, sizeof(int) * N,
    cudaMemcpyHostToDevice));

  add<<<N, 1>>>(device_v1, device_v2, device_v3);

  HANDLE_ERROR(cudaMemcpy(v3, device_v3, sizeof(int) * N,
    cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; i++)
    printf("%d + %d = %d\n", v1[i], v2[i], v3[i]);

  cudaFree(device_v1);
  cudaFree(device_v2);
  cudaFree(device_v3);

  return 0;
}
