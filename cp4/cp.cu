#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>

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

__global__ void compute_means(int ny, int nx, const float *data, float *means)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int y = tid; y < ny; y += stride)
  {
    float sum = 0.0f;
    for (int x = 0; x < nx; ++x)
    {
      sum += data[y * nx + x];
    }
    means[y] = sum / nx;
  }
}

__global__ void normalize_and_compute_pow_sums(int ny, int nx, const float *data, const float *means, float *norm_data, float *pow_sums)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int y = tid; y < ny; y += stride)
  {
    float pow_sum = 0.0f;
    for (int x = 0; x < nx; ++x)
    {
      float norm = data[y * nx + x] - means[y];
      norm_data[y * nx + x] = norm;
      pow_sum += norm * norm;
    }
    pow_sums[y] = pow_sum;
  }
}

__global__ void compute_correlations(int ny, int nx, const float *norm_data, const float *pow_sums, float *result)
{
  int i = blockIdx.x;
  int j = threadIdx.x;

  if (i < ny && j < i)
  {
    float sum = 0.0f;
    for (int x = 0; x < nx; ++x)
    {
      sum += norm_data[i * nx + x] * norm_data[j * nx + x];
    }
    float inv_sqrt_i = rsqrtf(pow_sums[i]);
    float inv_sqrt_j = rsqrtf(pow_sums[j]);
    result[i * ny + j] = sum * inv_sqrt_i * inv_sqrt_j;
  }
}

void correlate(int ny, int nx, const float *data, float *result)
{
  // Allocate GPU memory
  float *d_data, *d_means, *d_norm_data, *d_pow_sums, *d_result;

  size_t data_size = ny * nx * sizeof(float);
  size_t result_size = ny * ny * sizeof(float);

  CHECK(cudaMalloc((void **)&d_data, data_size));
  CHECK(cudaMalloc((void **)&d_means, ny * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_norm_data, data_size));
  CHECK(cudaMalloc((void **)&d_pow_sums, ny * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_result, result_size));

  // Copy data
  CHECK(cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));

  int block_size = 256;
  int grid_size = (ny + block_size - 1) / block_size;

  compute_means<<<grid_size, block_size>>>(ny, nx, d_data, d_means);
  CHECK(cudaGetLastError());

  normalize_and_compute_pow_sums<<<grid_size, block_size>>>(ny, nx, d_data, d_means, d_norm_data, d_pow_sums);
  CHECK(cudaGetLastError());

  dim3 grid_dim(ny, 1, 1);
  dim3 block_dim(ny, 1, 1);
  compute_correlations<<<grid_dim, block_dim>>>(ny, nx, d_norm_data, d_pow_sums, d_result);
  CHECK(cudaGetLastError());

  // Get results
  CHECK(cudaMemcpy(result, d_result, result_size, cudaMemcpyDeviceToHost));

  // Free GPU memory
  CHECK(cudaFree(d_data));
  CHECK(cudaFree(d_means));
  CHECK(cudaFree(d_norm_data));
  CHECK(cudaFree(d_pow_sums));
  CHECK(cudaFree(d_result));
}