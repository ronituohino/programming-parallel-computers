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

using namespace std;

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

void correlate(int ny, int nx, const float *data, float *result)
{
  vector<double> normal(nx * ny);
  vector<double> normal_square_sums(ny, 0.0);

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
    normal_square_sums[y] = sqrt(pow_sum);
  }

  // Apply padding to make matrix width a multiple of 'slice_len'
  constexpr int vector_size = 4;
  // slice_len has to be a multiple of vector_size
  constexpr int slice_len = 64;

  int vectors_per_slice = slice_len / vector_size;

  // parts is the amount of slices per row
  int parts = (nx + slice_len - 1) / slice_len;
  int nxp = parts * slice_len;

  int vectors_per_row = parts * vectors_per_slice;

  vector<double> padded(nxp * ny);
#pragma omp parallel for
  for (int y = 0; y < ny; y++)
  {
    for (int x = 0; x < nxp; x++)
    {
      if (x < nx)
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
  vector<double4_t> vectorized(ny * parts * vectors_per_slice);
#pragma omp parallel for
  for (int y = 0; y < ny; y++)
  {
    for (int p = 0; p < parts; p++)
    {
      for (int s = 0; s < slice_len / vector_size; s++)
      {
        for (int v = 0; v < vector_size; v++)
        {
          vectorized[y * vectors_per_row + p * vectors_per_slice + s][v] = padded[y * nxp + p * slice_len + (s * vector_size + v)];
        }
      }
    }
  }

// Calculate Pearson's correlation coefficient between all rows
#pragma omp parallel for
  for (int j = 0; j < ny; j++)
  {
    for (int i = ny - 1; i > j; i--)
    {
      vector<double4_t> sums(vectors_per_slice);
      for (int p = 0; p < parts; p++)
      {
        for (int v = 0; v < vectors_per_slice; v++)
        {
          sums[v] += vectorized[i * vectors_per_row + p * vectors_per_slice + v] * vectorized[j * vectors_per_row + p * vectors_per_slice + v];
        }
      }

      double4_t sum = {0.0, 0.0, 0.0, 0.0};
      for (int v = 0; v < vectors_per_slice; v++)
      {
        sum += sums[v];
      }
      double r = ((sum[0] + sum[1]) + (sum[2] + sum[3])) / (normal_square_sums[i] * normal_square_sums[j]);
      result[i + j * ny] = (float)r;
    }
  }

#pragma omp parallel for
  for (int n = 0; n < ny; n++)
  {
    result[n + n * ny] = 1.0;
  }
}