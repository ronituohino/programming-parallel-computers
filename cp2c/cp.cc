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

using namespace std;

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

void correlate(int ny, int nx, const float *data, float *result)
{
  vector<double> normal(nx * ny);
  vector<double> normal_square_sums(ny, 0.0);

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

  // Apply padding to make matrix width a multiple of 'slices'
  constexpr int slices = 4;
  int parts = (nx + slices - 1) / slices;
  int nxp = parts * slices;

  vector<double> padded(nxp * ny);
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
  vector<double4_t> vectorized(ny * parts);
  for (int y = 0; y < ny; y++)
  {
    for (int p = 0; p < parts; p++)
    {
      for (int s = 0; s < slices; s++)
      {
        vectorized[y * parts + p][s] = padded[y * nxp + p * slices + s];
      }
    }
  }

  // Calculate Pearson's correlation coefficient between all rows
  for (int j = 0; j < ny; j++)
  {
    for (int i = j; i < ny; i++)
    {
      double4_t sum = {0.0, 0.0, 0.0, 0.0};
      for (int p = 0; p < parts; p++)
      {
        sum += vectorized[i * parts + p] * vectorized[j * parts + p];
      }

      double r = ((sum[0] + sum[1]) + (sum[2] + sum[3])) / (normal_square_sums[i] * normal_square_sums[j]);
      result[i + j * ny] = (float)r;
    }
  }
}