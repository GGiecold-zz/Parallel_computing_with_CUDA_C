/* Due to the dependence on the OpenGL and GLUT libraries
   it is recommended to compile any program linking the
   present header by mentioning those libraries last on
   the command line. Namely:
   
     $nvcc my_program.cu -o my_program.o -lGL -lGLU -lglut
*/


#ifndef __BITMAP_H__
#define __BITMAP_H__


#include "my_gl.h"


#define CHANNELS 4


struct Bitmap {
  int w, h;
  unsigned char *pixels;
  void *data;
  void (*exit)(void*);

  Bitmap(int width, int height, void *ptr = NULL) {
    pixels = new unsigned char[width * height * CHANNELS]();
    w = width;
    h = height;
    data = ptr;
  }

  ~Bitmap() {
    delete [] pixels;
  }

  long size(void) const {
    return w * h * CHANNELS;
  }

  unsigned char *pointer(void) const {
    return pixels;
  }

  static Bitmap** get_pointer(void) {
    static Bitmap *bmp;
    return &bmp;
  }

  static void key(unsigned char k, int w, int h) {
    switch (k) {
      case 27:
        Bitmap* bmp = *(get_pointer());
      
        if (bmp->data != NULL && bmp->exit != NULL)
	  bmp->exit(bmp->data);
      
        break;
    }
  }

  static void draw(void) {
    Bitmap* bmp = *(get_pointer());
    
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(bmp->w, bmp->h, GL_RGBA, GL_UNSIGNED_BYTE,
		 bmp->pixels);
    glFlush();
  }

  void display(void (*f)(void *) = NULL) {
    Bitmap** bmp = get_pointer();
    *bmp = this;
    
    exit = f;

    int a = 1;
    char *s = (char *) (void *) "";
    glutInit(&a, &s);
    
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
    
    glutInitWindowSize(w, h);
    glutCreateWindow("bitmap");
    
    glutKeyboardFunc(key);
    glutDisplayFunc(draw);
    
    glutMainLoop();
  }
};


#endif
