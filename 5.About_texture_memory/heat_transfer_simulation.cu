/* Display the animation of a simple heat transfer simulation featuring a few
   heat sources (dubbed hereafter "heaters") placed at fixed locations
   on a two-dimensional grid and at various constant temperatures.

   Compile as follows at the CLI:

     nvcc heat_transfer_simulation.cu -o img.o -lGL -lGLU -lglut
*/


#include <cuda.h>
#include <stdio.h>
#include "../Common/animate.h"
#include "../Common/errors.h"
#include "../Common/lib.h"  /* for the swap function and
                               float_to_color kernel used below
			    */


#define DIM 1024
#define MAX_TEMPERATURE 1.0f
#define MIN_TEMPERATURE 0.0001f
#define SPEED 0.2f
#define TIME_STEPS_PER_FRAME 50


struct Data {
  cudaEvent_t start, stop;
  float compute_time;

  AnimateBitmap *bitmap;
  unsigned char *device_bitmap;
  float count_frames;

  float *device_heaters;
  float *device_input;
  float *device_output;
};


void animate_display(Data *, int);
void animate_exit(Data *);
__global__ void maintain_heaters(const float *, float *);
__global__ void temperature_update(const float *, float *);


int main(void)
{
  Data data;
  
  AnimateBitmap bitmap(DIM, DIM, &data);
  data.bitmap = &bitmap;
  
  data.compute_time = 0.0f;
  data.count_frames = 0;
  
  HANDLE_ERROR(cudaEventCreate(&data.start));
  HANDLE_ERROR(cudaEventCreate(&data.stop));

  HANDLE_ERROR(cudaMalloc((void **) &data.device_bitmap, bitmap.size()));
  HANDLE_ERROR(cudaMalloc((void **) &data.device_heaters, bitmap.size()));
  HANDLE_ERROR(cudaMalloc((void **) &data.device_input, bitmap.size()));
  HANDLE_ERROR(cudaMalloc((void **) &data.device_output, bitmap.size()));
  
  float *heaters_grid = new float[bitmap.size()];
  for (int i = 0; i < DIM * DIM; i++) {
    heaters_grid[i] = 0.0f;

    int x = i % DIM;
    int y = i / DIM;

    if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
      heaters_grid[i] = MAX_TEMPERATURE;

    heaters_grid[100 + 100 * DIM] = (MIN_TEMPERATURE + MAX_TEMPERATURE) / 2;
    heaters_grid[100 + 700 * DIM] = MIN_TEMPERATURE;
    heaters_grid[300 + 300 * DIM] = MIN_TEMPERATURE;
    heaters_grid[700 + 200 * DIM] = MIN_TEMPERATURE;

    for (int k = 800; k < 900; k++) {
      for (int j = 400; j < 500; j++)
        heaters_grid[j + k * DIM] = MIN_TEMPERATURE;
    }
  }

  HANDLE_ERROR(cudaMemcpy(data.device_heaters, heaters_grid,
    bitmap.size(), cudaMemcpyHostToDevice));

  for (int j = 800; j < DIM; j++)
    for (int i = 0; i < 200; i++)
      heaters_grid[i + j * DIM] = MAX_TEMPERATURE;

  HANDLE_ERROR(cudaMemcpy(data.device_heaters, heaters_grid,
    bitmap.size(), cudaMemcpyHostToDevice));

  delete [] heaters_grid;

  bitmap.animate((void (*)(void *, int)) animate_display,
    (void (*)(void *)) animate_exit);

  return 0;
}


void animate_display(Data *data, int clock)
{
  float compute_time;
  
  HANDLE_ERROR(cudaEventRecord(data->start, 0));

  AnimateBitmap *bitmap = data->bitmap;

  dim3 blocks_per_grid(DIM / 16, DIM / 16);
  dim3 threads_per_block(16, 16);

  for (int i = 0; i < TIME_STEPS_PER_FRAME; i++) {
    maintain_heaters<<<blocks_per_grid, threads_per_block>>>(
      data->device_heaters, data->device_input);
    temperature_update<<<blocks_per_grid, threads_per_block>>>(
      data->device_input, data->device_output);
    swap(data->device_input, data->device_output);
  }

  float_to_color<<<blocks_per_grid, threads_per_block>>>(
    data->device_input, data->device_bitmap);
  
  HANDLE_ERROR(cudaMemcpy(bitmap->pointer(), data->device_bitmap,
    bitmap->size(), cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaEventRecord(data->stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(data->stop));
  HANDLE_ERROR(cudaEventElapsedTime(&compute_time,
    data->start, data->stop));
  data->compute_time += compute_time;
  
  ++data->count_frames;

  printf("Mean compute time per frame: %.1f ms\n",
    data->compute_time / data->count_frames);
}


void animate_exit(Data *data)
{
  HANDLE_ERROR(cudaFree(data->device_heaters));
  HANDLE_ERROR(cudaFree(data->device_input));
  HANDLE_ERROR(cudaFree(data->device_output));
  
  HANDLE_ERROR(cudaEventDestroy(data->start));
  HANDLE_ERROR(cudaEventDestroy(data->stop));
}


__global__ void maintain_heaters(
  const float *heaters_ptr, float *input_ptr)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  if (heaters_ptr[offset] != 0)
    input_ptr[offset] = heaters_ptr[offset];
}


__global__ void temperature_update(
  const float *input_ptr, float *output_ptr)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  int left_neighbor = offset - 1;
  if (x == 0)
    ++left_neighbor;
  int right_neighbor = offset + 1;
  if (x == DIM - 1)
    --right_neighbor;
  int upper_neighbor = offset - DIM;
  if (y == 0)
    upper_neighbor += DIM;
  int lower_neighbor = offset + DIM;
  if (y == DIM - 1)
    lower_neighbor -= DIM;

  float gradient = input_ptr[upper_neighbor] + input_ptr[lower_neighbor] +
    input_ptr[left_neighbor] + input_ptr[right_neighbor] -
      4 * input_ptr[offset];
  output_ptr[offset] = input_ptr[offset] + SPEED * gradient;
}
