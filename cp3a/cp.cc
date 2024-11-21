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

static inline double4_t swap1(double4_t x) { return double4_t{x[1], x[0], x[3], x[2]}; }
static inline double4_t swap2(double4_t x) { return double4_t{x[2], x[3], x[0], x[1]}; }

void correlate(int ny, int nx, const float *data, float *result)
{
  // Padding to make result matrix height a multiple of 'y_slices'
  constexpr int y_slices = 4;
  int y_parts = (ny + y_slices - 1) / y_slices;

  // Padding to make result matrix width a multiple of 'x_slices'
  constexpr int x_slices = 6;
  int x_parts = (nx + x_slices - 1) / x_slices;
  int nxp = x_parts * x_slices;

  // Vectorize matrix
  vector<double4_t> v(nxp * y_parts);
#pragma omp parallel for
  for (int y = 0; y < y_parts; y++)
  {
    for (int x = 0; x < nxp; x++)
    {
      for (int s = 0; s < y_slices; s++)
      {
        if (y * y_slices + s < ny && x < nx)
        {
          v[y * nxp + x][s] = data[y * nx * y_slices + x + s * nx];
        }
        else
        {
          v[y * nxp + x][s] = 0.0;
        }
      }
    }
  }

  // Fix this and integrate with vectorization
  vector<double4_t> n(nxp * y_parts);
  vector<double> inv_nss(ny);
#pragma omp parallel for
  for (int y = 0; y < y_parts; y++)
  {
    double4_t sum;
    for (int x = 0; x < nx; x++)
    {
      sum += v[y * nxp + x];
    }
    double4_t mean = {
        sum[0] / nx,
        sum[1] / nx,
        sum[2] / nx,
        sum[3] / nx,
    };

    double4_t pow_sum;
    for (int x = 0; x < nx; x++)
    {
      double4_t normalized = v[y * nxp + x] - mean;
      n[y * nxp + x] = normalized;
      pow_sum += normalized * normalized;
    }

    for (int r = 0; r < y_slices; r++)
    {
      int ydx = y * y_slices + r;
      if (ydx < ny)
      {
        inv_nss[ydx] = 1.0 / sqrt(pow_sum[r]);
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
          double4_t a0 = n[i * nxp + (x * x_slices + k)];
          double4_t b0 = n[j * nxp + (x * x_slices + k)];

          double4_t a1 = swap1(a0);
          double4_t b1 = swap2(b0);

          double4_t a0b0 = a0 * b0;
          double4_t a0b1 = a0 * b1;
          double4_t a1b0 = a1 * b0;
          double4_t a1b1 = a1 * b1;

          sums[k + 0 * x_slices] += a0b0[0];
          sums[k + 1 * x_slices] += a1b0[1];
          sums[k + 2 * x_slices] += a0b1[0];
          sums[k + 3 * x_slices] += a1b1[1];

          sums[k + 4 * x_slices] += a1b0[0];
          sums[k + 5 * x_slices] += a0b0[1];
          sums[k + 6 * x_slices] += a1b1[0];
          sums[k + 7 * x_slices] += a0b1[1];

          sums[k + 8 * x_slices] += a0b1[2];
          sums[k + 9 * x_slices] += a1b1[3];
          sums[k + 10 * x_slices] += a0b0[2];
          sums[k + 11 * x_slices] += a1b0[3];

          sums[k + 12 * x_slices] += a1b1[2];
          sums[k + 13 * x_slices] += a0b1[3];
          sums[k + 14 * x_slices] += a1b0[2];
          sums[k + 15 * x_slices] += a0b0[3];
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
          result[(js + ns) + ((is + nx) * ny)] = (sums[idx] + sums[idx + 1] + sums[idx + 2] + sums[idx + 3] + sums[idx + 4] + sums[idx + 5]) * (inv_nss[(is + nx)] * inv_nss[(js + ns)]);
        }
      }
    }
  }
}