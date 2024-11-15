/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <cmath>
#include <vector>
#include <iostream>
#include <x86intrin.h>

using namespace std;

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

static inline double4_t swap1(double4_t x) { return _mm256_permute_pd(x, 0b0101); }
static inline double4_t swap2(double4_t x)
{
  double4_t p = _mm256_permute_pd(x, 0b1010);
  return _mm256_permute2f128_pd(p, p, 0b00000001);
}

void correlate(int ny, int nx, const float *data, float *result)
{
  vector<double> normal(nx * ny);
  vector<double> inv_nss(ny, 0.0);
#pragma omp parallel for
  for (int y = 0; y < ny; y++)
  {
    double sum = 0.0;
    for (int x = 0; x < nx; x++)
    {
      sum += data[y * nx + x];
    }
    double mean = sum / nx;

    double pow_sum = 0.0;
    for (int x = 0; x < nx; x++)
    {
      double normalized = data[y * nx + x] - mean;
      normal[y * nx + x] = normalized;
      pow_sum += pow(normalized, 2);
    }
    inv_nss[y] = 1.0 / sqrt(pow_sum);
  }

  // Padding to make result matrix height a multiple of 'y_slices'
  constexpr int y_slices = 4;
  int y_parts = (ny + y_slices - 1) / y_slices;
  int nyp = y_parts * y_slices;

  // Padding to make result matrix width a multiple of 'x_slices'
  constexpr int x_slices = 5;
  int x_parts = (nx + x_slices - 1) / x_slices;
  int nxp = x_parts * x_slices;

  vector<double> padded(nxp * nyp);
#pragma omp parallel for
  for (int y = 0; y < nyp; y++)
  {
    for (int x = 0; x < nxp; x++)
    {
      if (y < ny && x < nx)
      {
        padded[y * nxp + x] = normal[y * nx + x];
      }
      else
      {
        padded[y * nxp + x] = 0.0;
      }
    }
  }

  // Vectorize matrix
  vector<double4_t> v(nxp * y_parts);
#pragma omp parallel for
  for (int y = 0; y < y_parts; y++)
  {
    for (int x = 0; x < nxp; x++)
    {
      for (int s = 0; s < y_slices; s++)
      {
        v[y * nxp + x][s] = padded[y * nxp * y_slices + x + s * nxp];
      }
    }
  }

  // Calculate Pearson's correlation coefficient between all rows
#pragma omp parallel for collapse(2)
  for (int i = 0; i < y_parts; i++)
  {
    for (int j = i; j < y_parts; j++)
    {
      vector<double> sums(x_slices * 16);

      for (int x = 0; x < nxp / x_slices; x++)
      {
        for (int k = 0; k < x_slices; k++)
        {
          double4_t a0 = v[i * nxp + (x * x_slices + k)];
          double4_t b0 = v[j * nxp + (x * x_slices + k)];

          double4_t ab00 = a0 * b0;
          double4_t ab01 = a0 * swap1(b0);
          double4_t ab12 = swap1(a0) * swap2(b0);
          double4_t ab20 = swap2(a0) * b0;

          sums[k + 0 * x_slices] += ab00[0];
          sums[k + 1 * x_slices] += ab01[0];
          sums[k + 2 * x_slices] += ab20[2];
          sums[k + 3 * x_slices] += ab12[1];
          sums[k + 4 * x_slices] += ab01[1];
          sums[k + 5 * x_slices] += ab00[1];
          sums[k + 6 * x_slices] += ab12[0];
          sums[k + 7 * x_slices] += ab20[3];
          sums[k + 8 * x_slices] += ab20[0];
          sums[k + 9 * x_slices] += ab12[3];
          sums[k + 10 * x_slices] += ab00[2];
          sums[k + 11 * x_slices] += ab01[2];
          sums[k + 12 * x_slices] += ab12[2];
          sums[k + 13 * x_slices] += ab20[1];
          sums[k + 14 * x_slices] += ab01[3];
          sums[k + 15 * x_slices] += ab00[3];
        }
      }

      int is = i * y_slices;
      int js = j * y_slices;

      for (int n = 0; n < 16; n++)
      {
        int nx = n / 4;
        int ns = n % 4;

        if (js + ns < ny && is + nx < ny)
        {
          int idx = n * x_slices;
          result[(js + ns) + ((is + nx) * ny)] = (sums[idx + 0] + sums[idx + 1] + sums[idx + 2] + sums[idx + 3] + sums[idx + 4]) * (inv_nss[(is + nx)] * inv_nss[(js + ns)]);
        }
      }
    }
  }
}