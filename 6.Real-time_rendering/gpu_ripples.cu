/* Similar to 3_On_threads/ripples.cu but perform the animation
   entirely on the GPU.
   
   Bypassing the CPU for graphic rendering is done via a GPUAnimateBitmap
   structure, a variant of the AnimateBitmap structure used in many
   of our previous examples.
   Those structures are defined in ../Common/gpu_animate.h
   and ../Common/animate.h, respectively.
   
   The bulk of the newly-introduced concepts on graphic interoperability
   and the use of a shared buffer between a GPU device and an OpenGL driver
   appear in ../Common/gpu_animate.h
   
   Compile as follows:

     $nvcc gpu_ripples.cu -o gpu_ripples.out -lGL -lGLU -lglut
*/


#include <cuda.h>
#include "../Common/errors.h"
#include "../Common/gpu_animate.h"


#define DIM 1024


__global__ void kernel(uchar4 *, int);
void animate(uchar4 *, void *, int);


int main(void)
{
  GPUAnimateBitmap bitmap(DIM, DIM, NULL);
  bitmap.animate((void (*)(uchar4 *, void *, int)) animate, NULL);

  return 0;
}


__global__ void kernel(uchar4 *pixels, int clock)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float scaled_x = x - DIM / 2;
  float scaled_y = y - DIM / 2;
  float dist = sqrtf(scaled_x * scaled_x + scaled_y * scaled_y);

  unsigned char grey = (unsigned char) (128.0f +
    127.0f * cos(dist / 10.0f - clock / 7.0f) / (dist / 10.0f + 1.0f));

  pixels[offset].x = grey;
  pixels[offset].y = grey;
  pixels[offset].z = grey;
  pixels[offset].w = 255;
}


void animate(uchar4 *pixels, void *, int clock)
{
  dim3 blocks_per_grid(DIM / 16, DIM / 16);
  dim3 threads_per_block(16, 16);
  kernel<<<blocks_per_grid, threads_per_block>>>(pixels, clock);
}
