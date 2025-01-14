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

void correlate(int ny, int nx, const float *data, float *result)
{
  vector<double> normal(nx * ny);
  vector<double> normal_square_sums(ny, 0.0);

#pragma omp parallel for schedule(dynamic, 1)
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

// Calculate Pearson's correlation coefficient between all rows
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = 0; j < ny; j++)
  {
    for (int i = j; i < ny; i++)
    {
      double sum = 0.0;
      for (int n = 0; n < nx; n++)
      {
        sum += normal[i * nx + n] * normal[j * nx + n];
      }

      double r = sum / (normal_square_sums[i] * normal_square_sums[j]);
      result[i + j * ny] = (float)r;
    }
  }
}