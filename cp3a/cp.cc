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

  // Vectorize matrix
  vector<double4_t> v(nx * y_parts);
  vector<double4_t> means(y_parts);

#pragma omp parallel for schedule(dynamic, 1)
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
    means[y] = mean;
  }

  vector<double4_t> pr(y_parts * y_parts * 4);    // Partial results between stripes
  vector<double4_t> d(cols_per_stripe * y_parts); // Data that is handled in each stripe
  vector<double4_t> pow_sums(y_parts);

  for (int stripe = 0; stripe < nx; stripe += cols_per_stripe)
  {
    int stripe_end = min(nx - stripe, cols_per_stripe);

// Load in data, and normalize
#pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < y_parts; y++)
    {
      for (int x = 0; x < stripe_end; x++)
      {
        double4_t norm = v[y * nx + x + stripe] - means[y];
        d[y * cols_per_stripe + x] = norm;
        pow_sums[y] += norm * norm;
      }
    };

    // Calculate Pearson's correlation coefficient between all rows
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < y_parts; i++)
    {
      for (int j = i; j < y_parts; j++)
      {
        double4_t sums[4] = {};

        for (int x = 0; x < stripe_end; x++)
        {
          // Add ILP?
          double4_t a0 = d[i * cols_per_stripe + x];
          double4_t b0 = d[j * cols_per_stripe + x];

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
  }

  vector<double> inv_nss(nyp);
#pragma omp parallel for schedule(dynamic, 1)
  for (int n = 0; n < y_parts; n++)
  {
    for (int r = 0; r < y_slices; r++)
    {
      inv_nss[n * y_slices + r] = 1.0 / sqrt(pow_sums[n][r]);
    }
  }

#pragma omp parallel for schedule(dynamic, 1)
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