#include <math.h>
#include "../Common/errors.h"


#define min(x, y) ((x) < (y) ? (x) : (y))
#define sum_of_squares(n) ((n) * ((n) + 1) * (2 * (n) + 1) / 6)

#define N 21 * 1024


const int threads_per_block = 256;
const int blocks_per_grid = min(32,
  (N + threads_per_block - 1) / threads_per_block);


__global__ void inner_product(float *, float *, float *);


int main(void)
{
  float *x, *device_x;
  x = new float[N]();
  HANDLE_ERROR(cudaMalloc((void **) &device_x,
    sizeof(float) * N));

  float *y, *device_y;
  y = new float[N]();
  HANDLE_ERROR(cudaMalloc((void **) &device_y,
    sizeof(float) * N));

  float *z, *device_z;
  z = new float[blocks_per_grid]();
  HANDLE_ERROR(cudaMalloc((void **) &device_z,
    sizeof(float) * blocks_per_grid));

  for (int i = 0; i < N; i++)
    x[i] = y[i] = i;

  HANDLE_ERROR(cudaMemcpy(device_x, x, sizeof(float) * N,
    cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(device_y, y, sizeof(float) * N,
    cudaMemcpyHostToDevice));

  inner_product<<<blocks_per_grid, threads_per_block>>>(
    device_x, device_y, device_z);

  HANDLE_ERROR(cudaMemcpy(z, device_z, sizeof(float) * blocks_per_grid,
    cudaMemcpyDeviceToHost));

  float result = 0.0;
  for (int i = 0; i < blocks_per_grid; i++)
    result += z[i];

  printf("GPU computation vs. pyramidal square number: %.2f vs. %.2f\n",
    result, sum_of_squares((float) N - 1));
    
  bool fail_check = fabs(result - sum_of_squares((float) N - 1)) < 1E-10;
  if (fail_check == true)
    printf("The two results agree!\n");
  else
    printf("The two results differ...\n");

  cudaFree(device_x);
  cudaFree(device_y);
  cudaFree(device_z);

  delete [] x;
  delete [] y;
  delete [] z;

  return 0;
}


__global__ void inner_product(float *x, float *y, float *z)
{
  // buffer of shared memory, handling a running sum
  // toward the inner product of vectors x and y,
  // this running sum being partial to each block
  __shared__ float cache[threads_per_block];

  int cache_idx = threadIdx.x;
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  
  float running_sum = 0.0;

  while (id < N) {
    running_sum += x[id] * y[id];
    id += blockDim.x * gridDim.x;
  }

  cache[cache_idx] = running_sum;
  __syncthreads();

  int i = blockDim.x / 2;
  while (i != 0) {
    if (cache_idx < i)
      cache[cache_idx] += cache[cache_idx + i];
    __syncthreads();
    i /= 2;
  }

  if (cache_idx == 0)
    z[blockIdx.x] = cache[0];
}
