/* Compile as follows:

     $nvcc shared_memory_bitmap.cu -o shared_memory_bitmap.o -lGL -lGLU -lglut
*/


#include <cuda.h>
#include <math.h>
#include "../Common/bitmap.h"
#include "../Common/errors.h"


#define DIM 1024
#define PI (float) acos(-1.0)


__global__ void kernel(unsigned char *);


int main(void)
{
  Bitmap bitmap(DIM, DIM);
  
  unsigned char *device_bitmap;
  HANDLE_ERROR(cudaMalloc((void **) &device_bitmap, bitmap.size()));

  dim3 blocks_per_grid(DIM / 16, DIM / 16);
  dim3 threads_per_block(16, 16);
  kernel<<<blocks_per_grid, threads_per_block>>>(device_bitmap);

  HANDLE_ERROR(cudaMemcpy(bitmap.pointer(), device_bitmap,
    bitmap.size(), cudaMemcpyDeviceToHost));

  bitmap.display();

  cudaFree(device_bitmap);

  return 0;
}


__global__ void kernel(unsigned char *ptr)
{
  __shared__ float cache[16][16];

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  cache[threadIdx.x][threadIdx.y] =
    255 * (sinf(2 * x * PI / 123.0f) + 1.0f) *
      (sinf(2 * y * PI / 123.0f) + 1.0f) / 4.0f;
  /* Compare the results with and without __syncthreads; the latter
     is incorrect and would result in a blurry image
  */
  __syncthreads(); 
  
  ptr[0 + offset * 4] = 0;
  ptr[1 + offset * 4] =
    cache[15 - threadIdx.x][15 - threadIdx.y];
  ptr[2 + offset * 4] = 0;
  ptr[3 + offset * 4] = 255;
}
