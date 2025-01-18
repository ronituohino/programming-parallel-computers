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

// Returns the minimum units (cast to int to floor) to fill a, with b -size parts
static inline int divup(int a, int b)
{
  return (a + b - 1) / b;
}

__global__ void compute_means_and_normalize(int ny, int nx, int nyp, int nxp, const float *data, float *norm, float *nss)
{
  int y = blockIdx.x * blockDim.x + threadIdx.x;

  if (y >= ny)
  {
    return;
  }

  float sum = 0.0;
  for (int x = 0; x < nx; x++)
  {
    sum += data[y * nx + x];
  }
  float mean = sum / nx;

  float pow_sum = 0.0;
  for (int x = 0; x < nx; x++)
  {
    float normalized = data[y * nx + x] - mean;
    norm[y * nxp + x] = normalized;
    pow_sum += normalized * normalized;
  }
  nss[y] = 1.0 / sqrt(pow_sum);
}

__global__ void compute_correlations(int ny, int nx, int nyp, int nxp, int y_parts, int x_parts, const float *norm, float *par_res)
{
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y;
  int y1 = blockIdx.z;

  if (y1 < y0 || x0 >= x_parts)
  {
    return;
  }

  float sums[16] = {0.0};

  for (int x = 0; x < nxp / x_parts; x++)
  {
    for (int n = 0; n < 16; n++)
    {
      int nx_idx = n / 4;
      int ns_idx = n % 4;

      int i = y0 * 4 + nx_idx;
      int j = y1 * 4 + ns_idx;

      if (i < ny && j < ny && (x + x0 * (nxp / x_parts)) < nx)
      {
        float a = norm[i * nxp + x + x0 * (nxp / x_parts)];
        float b = norm[j * nxp + x + x0 * (nxp / x_parts)];
        sums[n] += a * b;
      }
    }
  }

  // Write out sums to partial results
  int addr = (x0 * nyp * nyp) + (y0 * y_parts + y1) * 16;
  for (int n = 0; n < 16; n++)
  {
    par_res[addr + n] = sums[n];
  }
}

__global__ void compute_sums(int ny, int nx, int nyp, int nxp, int y_parts, int x_parts, const float *par_res, const float *nss, float *result)
{
  int y0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y1 = blockIdx.y * blockDim.y + threadIdx.y;

  if (y1 < y0 || y0 >= y_parts || y1 >= y_parts)
  {
    return;
  }

  float sums[16] = {0.0};

  for (int x0 = 0; x0 < x_parts; x0++)
  {
    int addr = (x0 * nyp * nyp) + (y0 * y_parts + y1) * 16;

    // Fetch all values from partial results and calulate total sum for these rows
    for (int n = 0; n < 16; n++)
    {
      sums[n] += par_res[addr + n];
    }
  }

  for (int n = 0; n < 16; n++)
  {
    int nx_idx = n / 4;
    int ns_idx = n % 4;

    int i = y0 * 4 + nx_idx;
    int j = y1 * 4 + ns_idx;

    if (i < ny && j < ny)
    {
      result[i * ny + j] = sums[n] * (nss[i] * nss[j]);
    }
  }
}

void correlate(int ny, int nx, const float *data, float *result)
{
  int y_parts = divup(ny, 4);
  int nyp = y_parts * 4;

  int x_parts = divup(nx, 512);
  int nxp = x_parts * 512;

  // Allocate device memory
  float *d_data, *d_norm, *d_nss, *d_par_res, *d_result;

  size_t data_size = ny * nx * sizeof(float);
  size_t norm_size = nyp * nxp * sizeof(float);
  size_t nss_size = ny * sizeof(float);
  size_t par_res_size = x_parts * nyp * nyp * sizeof(float);
  size_t result_size = ny * ny * sizeof(float);

  CHECK(cudaMalloc((void **)&d_data, data_size));
  CHECK(cudaMalloc((void **)&d_norm, norm_size));
  CHECK(cudaMalloc((void **)&d_nss, nss_size));
  CHECK(cudaMalloc((void **)&d_par_res, par_res_size));
  CHECK(cudaMalloc((void **)&d_result, result_size));

  CHECK(cudaMemset(d_data, 0, data_size));
  CHECK(cudaMemset(d_norm, 0, norm_size));
  CHECK(cudaMemset(d_nss, 0, nss_size));
  CHECK(cudaMemset(d_par_res, 0, par_res_size));
  CHECK(cudaMemset(d_result, 0, result_size));

  // Copy input data to device
  CHECK(cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));

  // Launch kernel to compute means and normalize data
  {
    dim3 grid(divup(nyp, 8));
    dim3 block(8);
    compute_means_and_normalize<<<grid, block>>>(ny, nx, nyp, nxp, d_data, d_norm, d_nss);
    CHECK(cudaGetLastError());
  }

  // Launch kernel to compute correlations
  {
    dim3 grid(divup(x_parts, 8), y_parts, y_parts);
    dim3 block(8);
    compute_correlations<<<grid, block>>>(ny, nx, nyp, nxp, y_parts, x_parts, d_norm, d_par_res);
    CHECK(cudaGetLastError());
  }

  // Launch kernel to compute final sums
  {
    dim3 grid(divup(y_parts, 8), divup(y_parts, 8));
    dim3 block(8, 8);
    compute_sums<<<grid, block>>>(ny, nx, nyp, nxp, y_parts, x_parts, d_par_res, d_nss, d_result);
    CHECK(cudaGetLastError());
  }

  // Copy result back to host
  CHECK(cudaMemcpy(result, d_result, result_size, cudaMemcpyDeviceToHost));

  // Free device memory
  CHECK(cudaFree(d_data));
  CHECK(cudaFree(d_norm));
  CHECK(cudaFree(d_nss));
  CHECK(cudaFree(d_par_res));
  CHECK(cudaFree(d_result));
}
