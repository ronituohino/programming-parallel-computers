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

typedef float float8_t __attribute__((vector_size(8 * sizeof(float))));

static inline float8_t swap1(float8_t x) { return float8_t{x[1], x[0], x[3], x[2], x[5], x[4], x[7], x[6]}; }
static inline float8_t swap2(float8_t x) { return float8_t{x[2], x[3], x[0], x[1], x[6], x[7], x[4], x[5]}; }
static inline float8_t swap4(float8_t x) { return float8_t{x[4], x[5], x[6], x[7], x[0], x[1], x[2], x[3]}; }

constexpr int l[64] = {
    0, 4, 3, 7, 1, 5, 2, 6,
    4, 0, 7, 3, 5, 1, 6, 2,
    3, 7, 0, 4, 2, 6, 1, 5,
    7, 3, 4, 0, 6, 2, 5, 1,
    1, 5, 2, 6, 0, 4, 3, 7,
    5, 1, 6, 2, 4, 0, 7, 3,
    2, 6, 1, 5, 3, 7, 0, 4,
    6, 2, 5, 1, 7, 3, 4, 0};

constexpr int r[64] = {
    0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1,
    2, 3, 2, 3, 2, 3, 2, 3,
    2, 3, 2, 3, 2, 3, 2, 3,
    4, 5, 4, 5, 4, 5, 4, 5,
    4, 5, 4, 5, 4, 5, 4, 5,
    6, 7, 6, 7, 6, 7, 6, 7,
    6, 7, 6, 7, 6, 7, 6, 7};

void correlate(int ny, int nx, const float *data, float *result)
{
  constexpr int cols_per_stripe = 100;

  // Padding to make result matrix height a multiple of 'y_slices'
  constexpr int y_slices = 8;
  int y_parts = (ny + y_slices - 1) / y_slices;
  int nyp = y_parts * y_slices;

  // Vectorize matrix
  vector<float8_t> v(nx * y_parts);
  vector<float8_t> means(y_parts);

#pragma omp parallel for
  for (int y = 0; y < y_parts; y++)
  {
    float8_t sum = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

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

    float d = 1.0 / nx;
    float8_t div = {d, d, d, d, d, d, d, d};
    float8_t mean = sum * div;
    means[y] = mean;
  }

  vector<float8_t> pr(y_parts * y_parts * 8);    // Partial results between stripes
  vector<float8_t> d(cols_per_stripe * y_parts); // Data that is handled in each stripe
  vector<float8_t> pow_sums(y_parts);

  for (int stripe = 0; stripe < nx; stripe += cols_per_stripe)
  {
    int stripe_end = min(nx - stripe, cols_per_stripe);

// Load in data, and normalize
#pragma omp parallel for
    for (int y = 0; y < y_parts; y++)
    {
      for (int x = 0; x < stripe_end; x++)
      {
        float8_t norm = v[y * nx + x + stripe] - means[y];
        d[y * cols_per_stripe + x] = norm;
        pow_sums[y] += norm * norm;
      }
    };

    // Calculate Pearson's correlation coefficient between all rows
#pragma omp parallel for collapse(2)
    for (int i = 0; i < y_parts; i++)
    {
      for (int j = i; j < y_parts; j++)
      {
        vector<float8_t> sums(8);

        for (int x = 0; x < stripe_end; x++)
        {
          float8_t a0 = d[i * cols_per_stripe + x];
          float8_t b0 = d[j * cols_per_stripe + x];

          float8_t a1 = swap1(a0);
          float8_t b1 = swap4(b0);
          float8_t b2 = swap2(b1);
          float8_t b3 = swap2(b0);

          sums[0] += a0 * b0; // a0b0
          sums[1] += a0 * b1; // a0b1
          sums[2] += a0 * b2; // a0b2
          sums[3] += a0 * b3; // a0b3

          sums[4] += a1 * b0; // a1b0
          sums[5] += a1 * b1; // a1b1
          sums[6] += a1 * b2; // a1b2
          sums[7] += a1 * b3; // a1b3
        }

        pr[(i * y_parts + j) * 8 + 0] += sums[0];
        pr[(i * y_parts + j) * 8 + 1] += sums[1];
        pr[(i * y_parts + j) * 8 + 2] += sums[2];
        pr[(i * y_parts + j) * 8 + 3] += sums[3];
        pr[(i * y_parts + j) * 8 + 4] += sums[4];
        pr[(i * y_parts + j) * 8 + 5] += sums[5];
        pr[(i * y_parts + j) * 8 + 6] += sums[6];
        pr[(i * y_parts + j) * 8 + 7] += sums[7];
      }
    }
  }

  vector<float> inv_nss(nyp);
#pragma omp parallel for
  for (int n = 0; n < y_parts; n++)
  {
    for (int r = 0; r < y_slices; r++)
    {
      inv_nss[n * y_slices + r] = 1.0 / sqrt(pow_sums[n][r]);
    }
  }

#pragma omp parallel for collapse(2)
  for (int i = 0; i < y_parts; i++)
  {
    for (int j = i; j < y_parts; j++)
    {
      vector<float8_t> sums = {pr[(i * y_parts + j) * 8 + 0],
                               pr[(i * y_parts + j) * 8 + 1],
                               pr[(i * y_parts + j) * 8 + 2],
                               pr[(i * y_parts + j) * 8 + 3],
                               pr[(i * y_parts + j) * 8 + 4],
                               pr[(i * y_parts + j) * 8 + 5],
                               pr[(i * y_parts + j) * 8 + 6],
                               pr[(i * y_parts + j) * 8 + 7]};

      int is = i * y_slices;
      int js = j * y_slices;

      for (int n = 0; n < 64; n++)
      {
        int nx = n / 8;
        int ns = n % 8;

        if (js + ns < ny && is + nx < ny)
        {
          result[((is + nx) * ny) + (js + ns)] = sums[l[n]][r[n]] * inv_nss[is + nx] * inv_nss[js + ns];
        }
      }
    }
  }
}