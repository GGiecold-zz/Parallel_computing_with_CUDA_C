#ifndef __LIB_H__
#define __LIB_H__


__device__ unsigned char value(float, float, int);
__global__ void float_to_color(const float *, unsigned char *);
__global__ void float_to_color(const float *, uchar4 *);


template<typename T> void swap(T& x, T& y)
{
  T tmp = x;
  x = y;
  y = tmp;
}


__device__ unsigned char value(float x, float y, int hue)
{
  if (hue > 360)
    hue -= 360;
  else if (hue < 0)
    hue += 360;

  if (hue < 60)
    return (unsigned char) (255 * (x + (y - x) * hue / 60));

  if (hue < 180)
    return (unsigned char) (255 * y);

  if (hue < 240)
    return (unsigned char) (255 * (x + (y - x) * (240 - hue) / 60));

  return (unsigned char) (255 * x);
}


__global__ void float_to_color(
  const float *float_ptr, unsigned char *color_ptr)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float l = float_ptr[offset];
  float hue = (180 + (int) (360 * l)) % 360;
  float s = 1;
  float m1, m2;

  if (l <= 0.5f)
    m2 = l * (1 + s);
  else
    m2 = l + s - l * s;

  m1 = 2 * l - m2;
  
  color_ptr[0 + offset * 4] = value(m1, m2, hue + 120);
  color_ptr[1 + offset * 4] = value(m1, m2, hue);
  color_ptr[2 + offset * 4] = value(m1, m2, hue - 120);
  color_ptr[3 + offset * 4] = 255;
}


__global__ void float_to_color(const float *float_ptr, uchar4 *uchar4_ptr)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float l = float_ptr[offset];
  float hue = (180 + (int) (360 * l)) % 360;
  float s = 1;
  float m1, m2;

  if (l <= 0.5f)
    m2 = l * (1 + s);
  else
    m2 = l + s - l * s;

  m1 = 2 * l - m2;
  
  uchar4_ptr[0 + offset * 4] = value(m1, m2, hue + 120);
  uchar4_ptr[1 + offset * 4] = value(m1, m2, hue);
  uchar4_ptr[2 + offset * 4] = value(m1, m2, hue - 120);
  uchar4_ptr[3 + offset * 4] = 255;
}


#endif
