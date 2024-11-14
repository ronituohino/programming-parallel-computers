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
        v[y * nx + x][s] = padded[y * nx * y_slices + x + s * nx];
      }
    }
  }

// Calculate Pearson's correlation coefficient between all rows
#pragma omp parallel for
  for (int j = 0; j < y_parts; j++)
  {
    for (int i = j; i < y_parts; i++)
    {
      double sums[28];

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

        sums[0] += a01[0];
        sums[1] += a02[0];
        sums[2] += a12[2];
        sums[3] += ab00[0];
        sums[4] += ab01[0];
        sums[5] += ab20[2];
        sums[6] += ab12[1];
        sums[7] += a12[0];
        sums[8] += a02[1];
        sums[9] += ab01[1];
        sums[10] += ab00[1];
        sums[11] += ab12[0];
        sums[12] += ab20[3];
        sums[13] += a01[2];
        sums[14] += ab20[0];
        sums[15] += ab12[3];
        sums[16] += ab00[2];
        sums[17] += ab01[2];
        sums[18] += ab12[2];
        sums[19] += ab20[1];
        sums[20] += ab01[3];
        sums[21] += ab00[3];
        sums[22] += b01[0];
        sums[23] += b02[0];
        sums[24] += b12[1];
        sums[25] += b12[0];
        sums[26] += b02[1];
        sums[27] += b01[2];
      }

      for (int k = 0; k < 7; k++)
      {
        for (int l = k + 1; l < 8; l++)
        {
          int n = (k * (15 - k)) / 2 + (l - k - 1); // 0, 1, 2, ... 27

          int ax = (k < 4 ? k + i * y_parts : k + j * y_parts - 4);
          int bx = (l < 4 ? l + i * y_parts : l + j * y_parts - 4);

          if (ax < ny && bx < ny)
          {
            if (ny == 5 && nx == 2)
            {
              cout << n << "<-n ax->" << ax << endl;
              cout << n << "<-n bx->" << bx << endl;
            }
            double r = sums[n] * (inv_normal_square_sums[ax] * inv_normal_square_sums[bx]);
            result[ax + bx * ny] = (float)r;
          }
        }
      }
    }
  }

  for (int n = 0; n < ny; n++)
  {
    result[n + n * ny] = 1.0;
  }
}