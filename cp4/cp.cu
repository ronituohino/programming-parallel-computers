#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#define CHECK(x) check(x, #x)
static inline void check(cudaError_t err, const char *context)
{
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA error: " << context << ": "
              << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

static inline int divup(int a, int b)
{
  return (a + b - 1) / b;
}

__global__ void compute_means_and_normalize(int ny, int nx, const float *data, float *normal, float *nss)
{
  int y = blockIdx.x * blockDim.x + threadIdx.x;

  if (y >= ny)
  {
    return;
  }

  float sum = 0.0;
  for (int x = 0; x < nx; ++x)
  {
    sum += data[y * nx + x];
  }
  float mean = sum / nx;

  float pow_sum = 0.0;
  for (int x = 0; x < nx; ++x)
  {
    float normalized = data[y * nx + x] - mean;
    normal[y * nx + x] = normalized;
    pow_sum += normalized * normalized;
  }
  nss[y] = sqrt(pow_sum);
}

__global__ void compute_correlations(int ny, int nx, const float *normal, const float *nss, float *result)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i >= ny || j >= ny || j < i)
    return;

  float sum = 0.0;
  for (int n = 0; n < nx; n++)
  {
    sum += normal[i * nx + n] * normal[j * nx + n];
  }

  result[j + i * ny] = sum / (nss[i] * nss[j]);
}

void correlate(int ny, int nx, const float *data, float *result)
{
  // Allocate device memory
  float *d_data, *d_normal, *d_nss, *d_result;

  size_t data_size = ny * nx * sizeof(float);
  size_t normal_size = ny * nx * sizeof(float);
  size_t result_size = ny * ny * sizeof(float);
  size_t nss_size = ny * sizeof(float);

  CHECK(cudaMalloc((void **)&d_data, data_size));
  CHECK(cudaMalloc((void **)&d_normal, normal_size));
  CHECK(cudaMalloc((void **)&d_nss, nss_size));
  CHECK(cudaMalloc((void **)&d_result, result_size));

  CHECK(cudaMemset(d_data, 0, data_size));
  CHECK(cudaMemset(d_normal, 0, normal_size));
  CHECK(cudaMemset(d_nss, 0, nss_size));
  CHECK(cudaMemset(d_result, 0, result_size));

  // Copy input data to device
  CHECK(cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));

  // Launch kernel to compute means and normalize data
  {
    dim3 grid_dim(divup(ny, 32));
    dim3 block_dim(32); // Make amount of threads a multiple of 32 (GPU warp size)
    compute_means_and_normalize<<<grid_dim, block_dim>>>(ny, nx, d_data, d_normal, d_nss);
    CHECK(cudaGetLastError());
  }

  // Launch kernel to compute correlations
  {
    dim3 grid_dim(divup(ny, 32), divup(ny, 32), 1);
    dim3 block_dim(32, 32, 1); // Make amount of threads a multiple of 32 (GPU warp size)
    compute_correlations<<<grid_dim, block_dim>>>(ny, nx, d_normal, d_nss, d_result);
    CHECK(cudaGetLastError());
  }

  // Copy result back to host
  CHECK(cudaMemcpy(result, d_result, result_size, cudaMemcpyDeviceToHost));

  // Free device memory
  CHECK(cudaFree(d_data));
  CHECK(cudaFree(d_normal));
  CHECK(cudaFree(d_nss));
  CHECK(cudaFree(d_result));
}
