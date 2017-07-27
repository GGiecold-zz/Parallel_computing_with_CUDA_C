/* An illustration of read-only, constant memory and the associated
   concepts of caching and of a so-called "warp" of threads executed
   in lockstep; CUDA C events for performance evaluation and tuning;
   along with a basic ray tracing algorithm.
   
   The code is mostly similar to that in
   ray_tracing_without_constant_memory.cu, except that an array
   of Sphere data structures is allocated in constant memory.
   This array is not passed to the ray tracing kernel.
   In this regard, the __constant__ keyword and cudaMemcpyToSymbol()
   function for copying from host memory to constant memory on the GPU
   are introduced.

   Compile via the following at the command-line interface:

     $nvcc ray_tracing_with_constant_memory.cu -o img.o -lGL -lGLU -lglut
*/


#include <cuda.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "../Common/bitmap.h"		  
#include "../Common/errors.h"


#define random(x) ((x) * rand() / RAND_MAX)
#define DIM 1024
#define N_SPHERES 11
#define INF HUGE_VAL


struct Sphere {
  float x, y, z, radius;
  float r, g, b;
  
  __device__ float hit(float pixel_x, float pixel_y, float *n) {
    float diff_x = pixel_x - x;
    float diff_y = pixel_y - y;
    float d_squared = diff_x * diff_x + diff_y * diff_y;
    
    if (d_squared < radius * radius) {
      float diff_z = sqrtf(radius * radius - d_squared);
      *n = diff_z / radius;
      
      return diff_z + z;
    }

    return -INF;
  }
};


__global__ void kernel(unsigned char *);


__constant__ Sphere device_spheres[N_SPHERES];


int main(void)
{
  srand(time(NULL));

  cudaEvent_t start, stop;
  float compute_time;
  
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventRecord(start, 0));
  HANDLE_ERROR(cudaEventCreate(&stop));

  Bitmap bitmap(DIM, DIM);
  
  unsigned char *device_bitmap;
  HANDLE_ERROR(cudaMalloc((void **) &device_bitmap, bitmap.size()));

  Sphere *spheres = new Sphere[N_SPHERES]();
  for (int i = 0; i < N_SPHERES; i++) {
    spheres[i].x = random(1000.0f) - 500;
    spheres[i].y = random(1000.0f) - 500;
    spheres[i].z = random(1000.0f) - 500;
    
    spheres[i].radius = random(100.0f) + 20;

    spheres[i].r = random(1.0f);
    spheres[i].g = random(1.0f);
    spheres[i].b = random(1.0f);
  }

  HANDLE_ERROR(cudaMemcpyToSymbol(device_spheres, spheres,
    sizeof(Sphere) * N_SPHERES));
  delete [] spheres;

  dim3 blocks_per_grid(DIM / 16, DIM / 16);
  dim3 threads_per_block(16, 16);
  kernel<<<blocks_per_grid, threads_per_block>>>(device_bitmap);

  HANDLE_ERROR(cudaMemcpy(bitmap.pointer(), device_bitmap, bitmap.size(),
    cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&compute_time, start, stop));

  printf("GPU compute time: %.1f ms\n", compute_time);

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  bitmap.display();

  cudaFree(device_bitmap);

  return 0;
}


__global__ void kernel(unsigned char *bitmap_ptr)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float pixel_x = x - DIM / 2;
  float pixel_y = y - DIM / 2;

  float r, g, b;
  r = g = b = 0.0f;
  float max_depth = -INF;

  for (int i = 0; i < N_SPHERES; i++) {
    float n;
    float z = device_spheres[i].hit(pixel_x, pixel_y, &n);
    
    if (z > max_depth) {
      max_depth = z;
      
      float scale = n;
      r = scale * device_spheres[i].r;
      g = scale * device_spheres[i].g;
      b = scale * device_spheres[i].b;
    }
  }

  bitmap_ptr[0 + offset * 4] = (int) (r * 255);
  bitmap_ptr[1 + offset * 4] = (int) (g * 255);
  bitmap_ptr[2 + offset * 4] = (int) (b * 255);
  bitmap_ptr[3 + offset * 4] = 255;
}
