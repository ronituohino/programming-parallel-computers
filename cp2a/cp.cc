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

  vector<double> trans(nx * ny);
  for (int y = 0; y < ny; y++)
  {
    for (int x = 0; x < nx; x++)
    {
      trans[x * ny + y] = normal[y * nx + x];
    }
  }

  vector<double> mult(ny * ny);
  for (int y = 0; y < ny; y++)
  {
    for (int x = 0; x < ny; x++)
    {
      for (int k = 0; k < nx; ++k)
      {
        mult[y * ny + x] += normal[y * nx + k] * trans[k * ny + x];
      }
    }
  }

  for (int y = 0; y < ny; y++)
  {
    for (int x = 0; x < ny; x++)
    {
      double norm_product = normal_square_sums[x] * normal_square_sums[y];
      result[y * ny + x] = mult[y * ny + x] / norm_product;
    }
  }
}
