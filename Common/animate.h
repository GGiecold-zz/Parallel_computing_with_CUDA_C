#ifndef __ANIMATE_H__
#define __ANIMATE_H__


#include "my_gl.h"


#define CHANNELS 4


struct AnimateBitmap {
  int w, h, drag_x, drag_y;
  unsigned char *pixels;
  void *data;

  void (*fAnimateBitmap)(void *, int) ;
  void (*exitAnimateBitmap)(void *);
  void (*dragAnimateBitmap)(void *, int, int, int, int);

  AnimateBitmap(int width, int height, void *ptr = NULL) {
    w = width;
    h = height;
    pixels = new unsigned char[width * height * CHANNELS];
    data = ptr;
    dragAnimateBitmap = NULL;
  }
  
  ~AnimateBitmap() {
    delete [] pixels;
  }

  long size(void) const {
    return w * h * CHANNELS;
  }

  unsigned char *pointer(void) const {
    return pixels;
  }

  static AnimateBitmap** get_pointer(void) {
    static AnimateBitmap *bmp;
    return &bmp;
  }

  void drag(void (*f)(void *, int, int, int, int)) {
    dragAnimateBitmap = f;
  }
  
  static void key(unsigned char k, int x, int y) {
    switch (k) {
    case 27:
      AnimateBitmap* bmp = *(get_pointer());
      bmp->exitAnimateBitmap(bmp->data);
      break;
    }
  }

  static void draw(void) {
    AnimateBitmap* bmp = *(get_pointer());
    
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(bmp->w, bmp->h, GL_RGBA, GL_UNSIGNED_BYTE, bmp->pixels);
    glutSwapBuffers();
  }

  static void mouse(int left_right, int up_down, int x, int y) {
    if (left_right == GLUT_LEFT_BUTTON) {
      AnimateBitmap* bmp = *(get_pointer());

      if (up_down == GLUT_DOWN) {
        bmp->drag_x = x;
	bmp->drag_y = y;
      } else if (up_down == GLUT_UP)
        bmp->dragAnimateBitmap(bmp->data,
	  bmp->drag_x, bmp->drag_y, x, y);
    }
  }

  static void idle(void) {
    static int clock = 1;

    AnimateBitmap* bmp = *(get_pointer());
    bmp->fAnimateBitmap(bmp->data, clock++);
    glutPostRedisplay();
  }

  void animate(void (*f)(void *, int), void (*exit)(void *)) {
    AnimateBitmap** bmp = get_pointer();
    *bmp = this;
    
    fAnimateBitmap = f;
    exitAnimateBitmap = exit;

    int a = 1;
    char *s = (char *) (void *) "";
    glutInit(&a, &s);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    
    glutInitWindowSize(w, h);
    glutCreateWindow("bitmap");
    
    glutKeyboardFunc(key);
    glutDisplayFunc(draw);
    if (dragAnimateBitmap != NULL)
      glutMouseFunc(mouse);
    glutIdleFunc(idle);

    glutMainLoop();
  }
};


#endif
