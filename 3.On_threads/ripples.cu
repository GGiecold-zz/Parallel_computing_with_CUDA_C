#include <cuda.h>
#include "../Common/animate.h"
#include "../Common/errors.h"


#define DIM 1024


struct Data {
  unsigned char *device_bitmap;
  AnimateBitmap *bitmap;
};


void make_frame(Data *, int);
void clean(Data *);
__global__ void kernel(unsigned char *, int);


int main(void)
{
  Data data;
  AnimateBitmap bitmap(DIM, DIM, &data);
  data.bitmap = &bitmap;

  HANDLE_ERROR(cudaMalloc((void **) &data.device_bitmap, bitmap.size()));

  bitmap.animate((void (*)(void *, int)) make_frame,
                 (void (*)(void *)) clean);

  return 0;
}


void make_frame(Data *data, int clock) {
  dim3 blocks(DIM/16, DIM/16);
  dim3 threads(16, 16);
  kernel<<<blocks, threads>>>(data->device_bitmap, clock);

  HANDLE_ERROR(cudaMemcpy(data->bitmap->pointer(),
                          data->device_bitmap,
                          data->bitmap->size(),
			  cudaMemcpyDeviceToHost));
}


void clean(Data *d) {
  cudaFree(d->device_bitmap);
}


__global__ void kernel(unsigned char *ptr, int clock) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float shift_x = x - DIM / 2;
  float shift_y = y - DIM / 2;
  float dist = sqrtf(shift_x * shift_x + shift_y * shift_y);

  unsigned char grey = (unsigned char) (123.0f +
    122.0f * cos(dist / 10.0f - clock / 7.0f) / (dist / 10.0f + 1.0f));

  ptr[0 + offset * 4] = grey;
  ptr[1 + offset * 4] = grey;
  ptr[2 + offset * 4] = grey;
  ptr[3 + offset * 4] = 255;
}
