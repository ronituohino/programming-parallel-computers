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

typedef double double8_t __attribute__((vector_size(8 * sizeof(double))));

static inline double8_t swap1(double8_t x) { return _mm512_permute_pd(x, 0b01010101); }
static inline double8_t swap2(double8_t x)
{
  return double8_t{x[2], x[3], x[0], x[1], x[6], x[7], x[4], x[5]};
}
static inline double8_t swap4(double8_t x)
{
  return double8_t{x[4], x[5], x[6], x[7], x[0], x[1], x[2], x[3]};
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
  constexpr int y_slices = 8;
  int y_parts = (ny + y_slices - 1) / y_slices;
  int nyp = y_parts * y_slices;

  // Padding to make result matrix width a multiple of 'x_slices'
  constexpr int x_slices = 1;
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
  vector<double8_t> v(nxp * y_parts);
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
      vector<double> sums(x_slices * 64);

      for (int x = 0; x < nxp / x_slices; x++)
      {
        for (int k = 0; k < x_slices; k++)
        {
          double8_t a0 = v[i * nxp + (x * x_slices + k)];
          double8_t b0 = v[j * nxp + (x * x_slices + k)];

          double8_t a1 = swap1(a0);
          double8_t b1 = swap4(b0);
          double8_t b2 = swap2(b1);
          double8_t b3 = swap2(b0);

          double8_t a0b0 = a0 * b0;
          double8_t a0b1 = a0 * b1;
          double8_t a0b2 = a0 * b2;
          double8_t a0b3 = a0 * b3;

          double8_t a1b0 = a1 * b0;
          double8_t a1b1 = a1 * b1;
          double8_t a1b2 = a1 * b2;
          double8_t a1b3 = a1 * b3;

          sums[k + 0 * x_slices] += a0b0[0];
          sums[k + 1 * x_slices] += a1b0[1];
          sums[k + 2 * x_slices] += a0b3[0];
          sums[k + 3 * x_slices] += a1b3[1];
          sums[k + 4 * x_slices] += a0b1[0];
          sums[k + 5 * x_slices] += a1b1[1];
          sums[k + 6 * x_slices] += a0b2[0];
          sums[k + 7 * x_slices] += a1b2[1];

          sums[k + 8 * x_slices] += a1b0[0];
          sums[k + 9 * x_slices] += a0b0[1];
          sums[k + 10 * x_slices] += a1b3[0];
          sums[k + 11 * x_slices] += a0b3[1];
          sums[k + 12 * x_slices] += a1b1[0];
          sums[k + 13 * x_slices] += a0b1[1];
          sums[k + 14 * x_slices] += a1b2[0];
          sums[k + 15 * x_slices] += a0b2[1];

          sums[k + 16 * x_slices] += a0b3[2];
          sums[k + 17 * x_slices] += a1b3[3];
          sums[k + 18 * x_slices] += a0b0[2];
          sums[k + 19 * x_slices] += a1b0[3];
          sums[k + 20 * x_slices] += a0b2[2];
          sums[k + 21 * x_slices] += a1b2[3];
          sums[k + 22 * x_slices] += a0b1[2];
          sums[k + 23 * x_slices] += a1b1[3];

          sums[k + 24 * x_slices] += a1b3[2];
          sums[k + 25 * x_slices] += a0b3[3];
          sums[k + 26 * x_slices] += a1b0[2];
          sums[k + 27 * x_slices] += a0b0[3];
          sums[k + 28 * x_slices] += a1b2[2];
          sums[k + 29 * x_slices] += a0b2[3];
          sums[k + 30 * x_slices] += a1b1[2];
          sums[k + 31 * x_slices] += a0b1[3];

          sums[k + 32 * x_slices] += a0b1[4];
          sums[k + 33 * x_slices] += a1b1[5];
          sums[k + 34 * x_slices] += a0b2[4];
          sums[k + 35 * x_slices] += a1b2[5];
          sums[k + 36 * x_slices] += a0b0[4];
          sums[k + 37 * x_slices] += a1b0[5];
          sums[k + 38 * x_slices] += a0b3[4];
          sums[k + 39 * x_slices] += a1b3[5];

          sums[k + 40 * x_slices] += a1b1[4];
          sums[k + 41 * x_slices] += a0b1[5];
          sums[k + 42 * x_slices] += a1b2[4];
          sums[k + 43 * x_slices] += a0b2[5];
          sums[k + 44 * x_slices] += a1b0[4];
          sums[k + 45 * x_slices] += a0b0[5];
          sums[k + 46 * x_slices] += a1b3[4];
          sums[k + 47 * x_slices] += a0b3[5];

          sums[k + 48 * x_slices] += a0b2[6];
          sums[k + 49 * x_slices] += a1b2[7];
          sums[k + 50 * x_slices] += a0b1[6];
          sums[k + 51 * x_slices] += a1b1[7];
          sums[k + 52 * x_slices] += a0b3[6];
          sums[k + 53 * x_slices] += a1b3[7];
          sums[k + 54 * x_slices] += a0b0[6];
          sums[k + 55 * x_slices] += a1b0[7];

          sums[k + 56 * x_slices] += a1b2[6];
          sums[k + 57 * x_slices] += a0b2[7];
          sums[k + 58 * x_slices] += a1b1[6];
          sums[k + 59 * x_slices] += a0b1[7];
          sums[k + 60 * x_slices] += a1b3[6];
          sums[k + 61 * x_slices] += a0b3[7];
          sums[k + 62 * x_slices] += a1b0[6];
          sums[k + 63 * x_slices] += a0b0[7];
        }
      }

      int is = i * y_slices;
      int js = j * y_slices;

      for (int n = 0; n < 64; n++)
      {
        int nx = n / 8;
        int ns = n % 8;

        if (js + ns < ny && is + nx < ny)
        {
          int idx = n * x_slices;
          result[(js + ns) + ((is + nx) * ny)] = (sums[idx]) * (inv_nss[(is + nx)] * inv_nss[(js + ns)]);
        }
      }
    }
  }
}