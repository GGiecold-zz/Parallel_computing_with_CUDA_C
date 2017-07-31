#ifndef __GPU_ANIMATE_H__
#define __GPU_ANIMATE_H__


#define CHANNELS 4
#define GL_GLEXT_PROTOTYPES


#include <cuda.h>
#include <cuda_gl_interop.h>
#include "my_gl.h"


struct GPUAnimateBitmap {
  int w, h, drag_x, drag_y;
  void *data;

  GLuint buffer_GL;
  cudaGraphicsResource *buffer_CUDA;

  void (*fGPUAnimateBitmap)(uchar4 *, void *, int);
  void (*exitGPUAnimateBitmap)(void *);
  void (*dragGPUAnimateBitmap)(void *, int, int, int, int);

  GPUAnimateBitmap(int width, int height, void *ptr) {
    w = width;
    h = height;
    data = ptr;
    dragGPUAnimateBitmap = NULL;

    cudaDeviceProp properties;
    int device_id;

    memset(&properties, 0, sizeof(cudaDeviceProp));
    properties.major = 1;
    properties.minor = 0;

    HANDLE_ERROR(cudaChooseDevice(&device_id, &properties));
    HANDLE_ERROR(cudaGLSetGLDevice(device_id));  // Use this device for
                                                 // OpenGL interoperations
    int c = 1;
    char *s = (char *) (void *) "";
    glutInit(&c, &s);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(width, height);
    glutCreateWindow("gpu_bitmap");

    glGenBuffers(1, &buffer_GL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer_GL);
    
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, CHANNELS * width * height,
      NULL, GL_DYNAMIC_DRAW_ARB);
    
    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&buffer_CUDA,
      buffer_GL, cudaGraphicsMapFlagsNone));        
  }

  ~GPUAnimateBitmap() {
    free_buffer();
  }

  long size(void) const {
    return w * h * CHANNELS;
  }

  static GPUAnimateBitmap** get_pointer(void) {
    static GPUAnimateBitmap *bitmap;
    return &bitmap;
  }
  
  void drag(void (*f)(void *, int, int, int, int)) {
    dragGPUAnimateBitmap = f;
  }

  static void key(unsigned char k, int x, int y) {
    switch (k) {
    case 27:
      GPUAnimateBitmap* bitmap = *(get_pointer());
      
      if (bitmap->exitGPUAnimateBitmap)
	bitmap->exitGPUAnimateBitmap(bitmap->data);

      bitmap->free_buffer();
      break;
    }
  }

  static void draw(void) {
    GPUAnimateBitmap* bitmap = *(get_pointer());

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glDrawPixels(bitmap->w, bitmap->h, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glutSwapBuffers();
  }

  static void mouse(int left_right, int up_down, int x, int y) {
    if (left_right == GLUT_LEFT_BUTTON) {
      GPUAnimateBitmap* bitmap = *(get_pointer());

      if (up_down == GLUT_DOWN) {
        bitmap->drag_x = x;
	bitmap->drag_y = y;
      } else if (up_down == GLUT_UP) {
        bitmap->dragGPUAnimateBitmap(bitmap->data,
	  bitmap->drag_x, bitmap->drag_y, x, y);
      }
    }
  }
  
  static void idle(void) {
    static int clock = 1;

    GPUAnimateBitmap* bitmap = *(get_pointer());
    
    uchar4 *device_ptr;
    size_t size;

    HANDLE_ERROR(cudaGraphicsMapResources(1, &(bitmap->buffer_CUDA), NULL));
    HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer(
      (void **) &device_ptr, &size, bitmap->buffer_CUDA));

    bitmap->fGPUAnimateBitmap(device_ptr, bitmap->data, clock++);
    // Unmaps the device pointer and release the shared buffer
    // for the OpenGL rendering
    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &(bitmap->buffer_CUDA), NULL));

    glutPostRedisplay();
  }
  
  void free_buffer(void) {
    HANDLE_ERROR(cudaGraphicsUnregisterResource(buffer_CUDA));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glDeleteBuffers(1, &buffer_GL);
  }

  void animate(void (*f)(uchar4 *, void *, int), void (*exit)(void *)) {
    GPUAnimateBitmap** bitmap = get_pointer();
    *bitmap = this;
    
    fGPUAnimateBitmap = f;
    exitGPUAnimateBitmap = exit;

    glutKeyboardFunc(key);
    glutDisplayFunc(draw);
    if (dragGPUAnimateBitmap != NULL)
      glutMouseFunc(mouse);
    glutIdleFunc(idle);

    glutMainLoop();
  }
};


#endif
