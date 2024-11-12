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
  vector<double> inv_normal_square_sums(ny, 0.0);
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
    inv_normal_square_sums[y] = 1.0 / sqrt(pow_sum);
  }

  // Apply padding to make matrix height a multiple of 'y_slices'
  constexpr int y_slices = 4;
  int y_parts = (ny + y_slices - 1) / y_slices;
  int nyp = y_parts * y_slices;

  vector<double> padded(nx * nyp);
#pragma omp parallel for
  for (int y = 0; y < nyp; y++)
  {
    for (int x = 0; x < nx; x++)
    {
      if (y < ny)
      {
        padded[y * nx + x] = normal[y * nx + x];
      }
      else
      {
        padded[y * nx + x] = 0.0;
      }
    }
  }

  // Vectorize matrix
  vector<double4_t> v(nx * y_parts);
#pragma omp parallel for

  for (int y = 0; y < y_parts; y++)
  {
    for (int x = 0; x < nx; x++)
    {
      for (int s = 0; s < y_slices; s++)
      {
        v[y * nx + x][s] = padded[y * nx * y_slices + x * y_slices + s];
      }
    }
  }

// Calculate Pearson's correlation coefficient between all rows
#pragma omp parallel for
  for (int j = 0; j < y_parts; j++)
  {
    for (int i = j; i < y_parts; i++)
    {
      int y_sp = pow(y_slices, 2);
      vector<double4_t> sums(y_sp);
      int ia = i * nx;
      int ja = j * nx;

      for (int x = 0; x < nx; x++)
      {
        for (int n = 0; n < y_sp; n++)
        {
          int ix = ia + n / y_slices; // 0, 0, 0, 1, 1, 1 ...
          int jx = ja + n % y_slices; // 0, 1, 2, 0, 1, 2 ...

          if (ix < ny && jx < ny)
          {
            sums[n] += v[ix * nx + x] * v[jx * nx + x];
          }
        }
      }

      for (int n = 0; n < y_sp; n++)
      {
        int ix = ia + n / y_slices;
        int jx = ja + n % y_slices;

        if (ix < ny && jx < ny)
        {
          double4_t s = sums[n];
          double r = ((s[0] + s[1]) + (s[2] + s[3])) * (inv_normal_square_sums[ix] * inv_normal_square_sums[jx]);
          result[ix + jx * ny] = r;
        }
      }
    }
  }
}