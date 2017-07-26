#include <stdio.h>

static void HandleError(cudaError_t, const char *, int);

#define HANDLE_ERROR(err) HandleError((err), __FILE__, __LINE__)
#define HANDLE_NULL(x) {if (x == NULL) { \
      printf("Host memory failure at line %d of %s\n", __LINE__, __FILE__); \
      exit(EXIT_FAILURE); \
    } \
  }

static void HandleError(cudaError_t err, const char *fname, int lino)
{
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), fname, lino);
    exit(EXIT_FAILURE);
  }
}
