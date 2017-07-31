/* Illustration of basic graphics interoperations and shared data buffers
   between OpenGL rendering and CUDA kernels. Similar procedures exist
   for DirectX interoperations.

   Compile at the CLI via:

     $nvcc graphics_interoperations_101.cu -o a.out -lGL -lGLU -lglut
*/


#define GL_GLEXT_PROTOTYPES


#include <GL/glut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include "../Common/animate.h"
#include "../Common/errors.h"


#define DIM 512


// GL and CUDA access a shared data buffer via variables of distinct types
GLuint buffer_GL;
cudaGraphicsResource *buffer_CUDA;


__global__ void kernel(uchar4 *);
static void keyboard_function(unsigned char, int, int);
static void display_function(void);


int main(int argc, char *argv[])
{
  int device_id;
  cudaDeviceProp properties;

  memset(&properties, 0, sizeof(cudaDeviceProp));
  properties.major = 1;
  properties.minor = 0;

  HANDLE_ERROR(cudaChooseDevice(&device_id, &properties));
  HANDLE_ERROR(cudaGLSetGLDevice(device_id));

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(DIM, DIM);
  glutCreateWindow("bitmap");

  glGenBuffers(1, &buffer_GL);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer_GL);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, 4 * DIM * DIM, NULL,
    GL_DYNAMIC_DRAW_ARB);

  HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&buffer_CUDA, buffer_GL,
    cudaGraphicsMapFlagsNone));

  uchar4 *device_ptr;
  size_t size;
  
  HANDLE_ERROR(cudaGraphicsMapResources(1, &buffer_CUDA, NULL));
  HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer(
    (void **) &device_ptr, &size, buffer_CUDA));

  dim3 blocks_per_grid(DIM / 16, DIM / 16);
  dim3 threads_per_block(16, 16);

  kernel<<<blocks_per_grid, threads_per_block>>>(device_ptr);

  // Synchronizes between CUDA computations and OpenGL graphics rendering
  HANDLE_ERROR(cudaGraphicsUnmapResources(1, &buffer_CUDA, NULL));
  
  glutKeyboardFunc(keyboard_function);
  glutDisplayFunc(display_function);
  glutMainLoop();

  return 0;
}


__global__ void kernel(uchar4 *ptr)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float normalized_x = x / (float) DIM - 0.5f;
  float normalized_y = y / (float) DIM - 0.5f;

  unsigned char green = 128 + 127 *
    sin(abs(100 * normalized_x) - abs(100 * normalized_y));

  ptr[offset].x = 0;
  ptr[offset].y = green;
  ptr[offset].z = 0;
  ptr[offset].w = 255;
}


static void keyboard_function(unsigned char k, int x, int y)
{
  switch(k) {
  case 27:
    HANDLE_ERROR(cudaGraphicsUnregisterResource(buffer_CUDA));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glDeleteBuffers(1, &buffer_GL);
    break;
  }
}


static void display_function(void)
{
  glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glutSwapBuffers();
}
