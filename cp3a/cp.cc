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

  for (int n = 0; n < ny; n++)
  {
    result[n + n * ny] = 1.0;
  }

  // Padding to make result matrix height a multiple of 'y_slices'
  constexpr int y_slices = 4;
  int y_parts = (ny + y_slices - 1) / y_slices;
  int nyp = y_parts * y_slices;

  // For small input, give result in a naive way
  if (y_parts < 2)
  {
    for (int i = 0; i < ny - 1; i++)
    {
      for (int j = i + 1; j < ny; j++)
      {
        double sum = 0.0;
        for (int x = 0; x < nx; x++)
        {
          sum += normal[i * nx + x] * normal[j * nx + x];
        }
        result[j + i * ny] = (float)(sum * (inv_nss[i] * inv_nss[j]));
      }
    }
    return;
  }

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
        v[y * nx + x][s] = padded[y * nx * y_slices + x + s * nx];
      }
    }
  }

  // Calculate Pearson's correlation coefficient between all rows
#pragma omp parallel for
  for (int i = 0; i < y_parts - 1; i++)
  {
    for (int j = i + 1; j < y_parts; j++)
    {
      vector<double> sums(28);

      for (int x = 0; x < nx; x++)
      {
        double4_t a0 = v[i * nx + x];
        double4_t b0 = v[j * nx + x];

        double4_t a1 = swap1(a0);
        double4_t a2 = swap2(a0);

        double4_t b1 = swap1(b0);
        double4_t b2 = swap2(b0);

        double4_t ab00 = a0 * b0;
        double4_t ab01 = a0 * b1;
        double4_t ab12 = a1 * b2;
        double4_t ab20 = a2 * b0;

        double4_t a01 = a0 * a1;
        double4_t a02 = a0 * a2;
        double4_t a12 = a1 * a2;

        double4_t b01 = b0 * b1;
        double4_t b02 = b0 * b2;
        double4_t b12 = b1 * b2;

        // Crossing results
        sums[0] += ab00[0];
        sums[1] += ab01[0];
        sums[2] += ab20[2];
        sums[3] += ab12[1];

        sums[4] += ab01[1];
        sums[5] += ab00[1];
        sums[6] += ab12[0];
        sums[7] += ab20[3];

        sums[8] += ab20[0];
        sums[9] += ab12[3];
        sums[10] += ab00[2];
        sums[11] += ab01[2];

        sums[12] += ab12[2];
        sums[13] += ab20[1];
        sums[14] += ab01[3];
        sums[15] += ab00[3];

        // Top left (self a0)
        sums[16] += a01[0];
        sums[17] += a02[0];
        sums[18] += a12[2];
        sums[19] += a12[0];
        sums[20] += a02[1];
        sums[21] += a01[2];

        // Bottom right (self b0)
        sums[22] += b01[0];
        sums[23] += b02[0];
        sums[24] += b12[1];
        sums[25] += b12[0];
        sums[26] += b02[1];
        sums[27] += b01[2];
      }

      int is = i * y_slices;
      int js = j * y_slices;

      // Top left of top right triangle
      for (int n = 16; n < 22; n++)
      {
        int nx = (n - 16) / 3 + (n == 21 ? 1 : 0); // 0, 0, 0, 1, 1, 2
        int ns = 1 + (n < 19 ? n - 16 : (n == 19) ? 1
                                                  : 2); // 1, 2, 3, 2, 3, 3

        if (is + ns < ny && is + nx < ny)
        {
          result[(is + ns) + ((is + nx) * ny)] = sums[n] * (inv_nss[(is + nx)] * inv_nss[(is + ns)]);
        }
      }

      // Crossing results
      for (int n = 0; n < 16; n++)
      {
        int nx = n / 4;
        int ns = n % 4;

        if (js + ns < ny && is + nx < ny)
        {
          result[(js + ns) + ((is + nx) * ny)] = sums[n] * (inv_nss[(is + nx)] * inv_nss[(js + ns)]);
        }
      }

      // Bottom right of top right triangle
      for (int n = 22; n < 28; n++)
      {
        int nx = (n - 22) / 3 + (n == 27 ? 1 : 0); // 0, 0, 0, 1, 1, 2
        int ns = 1 + (n < 25 ? n - 22 : (n == 25) ? 1
                                                  : 2); // 1, 2, 3, 2, 3, 3

        if (js + ns < ny && js + nx < ny)
        {
          result[(js + ns) + ((js + nx) * ny)] = sums[n] * (inv_nss[(js + nx)] * inv_nss[(js + ns)]);
        }
      }
    }
  }
}