/* Due to the bitmap.h header linked herewith being dependent
   on the OpenGL and GLUT libraries, please compile this program
   by mentioning those libraries last on the command line, namely

     $nvcc fractals.cu -o fractals.o -lGL -lGLU -lglut
*/

#include <assert.h>
#include <cuComplex.h>
/* The source code for this header can be found at

     http://pixel.ecn.purdue.edu:8080/purpl/WSJ/projects/ \
       DirectionalStippling/include/cuComplex.h
       
   Alternatively, with CUDA 7 or higher -- to check your version,
   type in $nvcc --version at the CLI -- we could have used
   
     struct thrust::complex<T>
     
   from the Thrust parallel algorithms library
*/
#include <stdio.h>
#include "../Common/bitmap.h"
#include "../Common/errors.h"


#define DIM 1024


__global__ void kernel(unsigned char *);
__device__ bool is_in_julia_set(int, int, const float);


int main(void)
{
  Bitmap bitmap(DIM, DIM);
  unsigned char *device_bitmap;

  HANDLE_ERROR(cudaMalloc((void **) &device_bitmap, bitmap.size()));

  dim3 grid(DIM, DIM);
  kernel<<<grid, 1>>>(device_bitmap);

  HANDLE_ERROR(cudaMemcpy(bitmap.pointer(), device_bitmap, bitmap.size(),
    cudaMemcpyDeviceToHost));

  bitmap.display();

  cudaFree(device_bitmap);

  return 0;
}


__global__ void kernel(unsigned char* p)
{
  int id_x = blockIdx.x;
  int id_y = blockIdx.y;
  
  int offset = id_x + id_y * gridDim.x;
  int is_julia = (int) is_in_julia_set(id_x, id_y, 1.0);
  
  p[0 + offset * 4] = is_julia * 255;
  p[1 + offset * 4] = 0;
  p[2 + offset * 4] = 0;
  p[3 + offset * 4] = 255;
}


__device__ bool is_in_julia_set(int x, int y, const float scale)
{
  assert (scale > 0);
  
  /* Shift by DIM/2 to center the complex plane at the image center
     then divide by DIM / 2 for the image to range from -1 to 1.
  */
  float normalized_x = scale * (float)(DIM / 2 - x) / (DIM / 2);
  float normalized_y = scale * (float)(DIM / 2 - y) / (DIM / 2);

  cuFloatComplex c = make_cuFloatComplex(-0.789, 0.123);
  cuFloatComplex z = make_cuFloatComplex(normalized_x, normalized_y);

  for (int i = 0; i < 100; i++) {
    /* For any complex constant c, a point is part of the Julia set
       if the sequence of numbers it generates through iteration
       of the relation z_{n+1} = z_{n} ** 2 + c
       are of bounded norm.
    */
    z = cuCmulf(z, z);
    z = cuCaddf(z, c);

    if (cuCabsf(z) > 1000)
      return false;
  }

  return true;
}
