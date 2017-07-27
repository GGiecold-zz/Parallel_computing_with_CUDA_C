/* An illustration of CUDA C events for performance evaluation
   and tuning, along with a basic ray tracing algorithm.

   Compile via the following at the command-line interface:

     $nvcc ray_tracing_without_constant_memory.cu -o img.o -lGL -lGLU -lglut
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


__global__ void kernel(Sphere *, unsigned char *);


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

  Sphere *device_spheres;
  HANDLE_ERROR(cudaMalloc((void **) &device_spheres,
    sizeof(Sphere) * N_SPHERES));

  HANDLE_ERROR(cudaMemcpy(device_spheres, spheres,
    sizeof(Sphere) * N_SPHERES, cudaMemcpyHostToDevice));
  delete [] spheres;

  dim3 blocks_per_grid(DIM / 16, DIM / 16);
  dim3 threads_per_block(16, 16);
  kernel<<<blocks_per_grid, threads_per_block>>>(device_spheres, device_bitmap);

  HANDLE_ERROR(cudaMemcpy(bitmap.pointer(), device_bitmap, bitmap.size(),
    cudaMemcpyDeviceToHost));
   
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&compute_time, start, stop));

  printf("GPU compute time: %.1f ms\n", compute_time);

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  bitmap.display();

  cudaFree(device_spheres);
  cudaFree(device_bitmap);

  return 0;
}


__global__ void kernel(Sphere *sphere_ptr, unsigned char *bitmap_ptr)
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
    float z = sphere_ptr[i].hit(pixel_x, pixel_y, &n);
    
    if (z > max_depth) {
      max_depth = z;
      
      float scale = n;
      r = scale * sphere_ptr[i].r;
      g = scale * sphere_ptr[i].g;
      b = scale * sphere_ptr[i].b;
    }
  }

  bitmap_ptr[0 + offset * 4] = (int) (r * 255);
  bitmap_ptr[1 + offset * 4] = (int) (g * 255);
  bitmap_ptr[2 + offset * 4] = (int) (b * 255);
  bitmap_ptr[3 + offset * 4] = 255;
}
