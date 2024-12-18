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
#include <tuple>
#include <iostream>
#include <x86intrin.h>

using namespace std;

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

static inline double4_t swap1(double4_t x) { return double4_t{x[1], x[0], x[3], x[2]}; }
static inline double4_t swap2(double4_t x) { return double4_t{x[2], x[3], x[0], x[1]}; }

constexpr int l[16] = {
    0, 2, 1, 3,
    2, 0, 3, 1,
    1, 3, 0, 2,
    3, 1, 2, 0};

constexpr int r[16] = {
    0, 1, 0, 1,
    0, 1, 0, 1,
    2, 3, 2, 3,
    2, 3, 2, 3};

void correlate(int ny, int nx, const float *data, float *result)
{
  constexpr int cols_per_stripe = 100;

  // Padding to make result matrix height a multiple of 'y_slices'
  constexpr int y_slices = 4;
  int y_parts = (ny + y_slices - 1) / y_slices;
  int nyp = y_parts * y_slices;

  // Build a Z-order curve memory access pattern
  std::vector<std::tuple<int, int, int>> rows(y_parts * y_parts);
#pragma omp parallel for
  for (int i = 0; i < y_parts; i++)
  {
    for (int j = 0; j < y_parts; j++)
    {
      int ij = _pdep_u32(i, 0x55555555) | _pdep_u32(j, 0xAAAAAAAA);
      rows[i * y_parts + j] = make_tuple(ij, i, j);
    }
  }

  // Vectorize matrix
  vector<double4_t> v(nx * y_parts);
  vector<double> inv_nss(nyp);

#pragma omp parallel for
  for (int y = 0; y < y_parts; y++)
  {
    double4_t sum = {0.0, 0.0, 0.0, 0.0};

    for (int x = 0; x < nx; x++)
    {
      for (int s = 0; s < y_slices; s++)
      {
        if (y * y_slices + s < ny && x < nx)
        {
          v[y * nx + x][s] = data[(y * y_slices + s) * nx + x];
        }
        else
        {
          v[y * nx + x][s] = 0.0;
        }
      }

      sum += v[y * nx + x];
    }

    double d = 1.0 / nx;
    double4_t div = {d, d, d, d};
    double4_t mean = sum * div;

    double4_t pow_sum = {0.0, 0.0, 0.0, 0.0};
    for (int x = 0; x < nx; x++)
    {
      double4_t normalized = v[y * nx + x] - mean;
      v[y * nx + x] = normalized;
      pow_sum += normalized * normalized;
    }

    for (int r = 0; r < y_slices; r++)
    {
      inv_nss[y * y_slices + r] = 1.0 / sqrt(pow_sum[r]);
    }
  }

  vector<double4_t> pr(y_parts * y_parts * 4);
  for (int stripe = 0; stripe < nx; stripe += cols_per_stripe)
  {
    int stripe_end = stripe + min(nx - stripe, cols_per_stripe);
    // Calculate Pearson's correlation coefficient between all rows

#pragma omp parallel for
    for (int n = 0; n < y_parts * y_parts; n++)
    {
      // Get corresponding pair (x, y) for Z-order value
      int ij, i, j;
      std::tie(ij, i, j) = rows[n];
      (void)ij;

      if (i > j)
      {
        continue;
      }

      vector<double4_t> sums(4);

      for (int x = stripe; x < stripe_end; x++)
      {
        double4_t a0 = v[i * nx + x];
        double4_t b0 = v[j * nx + x];

        double4_t a1 = swap1(a0);
        double4_t b1 = swap2(b0);

        sums[0] += a0 * b0; // a0b0
        sums[1] += a0 * b1; // a0b1
        sums[2] += a1 * b0; // a1b0
        sums[3] += a1 * b1; // a1b1
      }

      pr[(i * y_parts + j) * 4 + 0] += sums[0];
      pr[(i * y_parts + j) * 4 + 1] += sums[1];
      pr[(i * y_parts + j) * 4 + 2] += sums[2];
      pr[(i * y_parts + j) * 4 + 3] += sums[3];
    }
  }

#pragma omp parallel for collapse(2)
  for (int i = 0; i < y_parts; i++)
  {
    for (int j = i; j < y_parts; j++)
    {
      vector<double4_t> sums = {pr[(i * y_parts + j) * 4 + 0],
                                pr[(i * y_parts + j) * 4 + 1],
                                pr[(i * y_parts + j) * 4 + 2],
                                pr[(i * y_parts + j) * 4 + 3]};

      int is = i * y_slices;
      int js = j * y_slices;

      for (int n = 0; n < 16; n++)
      {
        int nx = n / 4;
        int ns = n % 4;

        if (js + ns < ny && is + nx < ny)
        {
          result[((is + nx) * ny) + (js + ns)] = sums[l[n]][r[n]] * inv_nss[is + nx] * inv_nss[js + ns];
        }
      }
    }
  }
}