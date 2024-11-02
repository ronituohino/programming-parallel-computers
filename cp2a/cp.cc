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

    double factor = sqrt(pow_sum);
    for (int x = 0; x < nx; x++)
    {
      normal[y * nx + x] /= factor;
    }
  }

  vector<double> trans(nx * ny);
  for (int y = 0; y < ny; y++)
  {
    for (int x = 0; x < nx; x++)
    {
      trans[x * ny + y] = normal[y * nx + x];
    }
  }

  for (int y = 0; y < ny; y++)
  {
    for (int x = 0; x < ny; x++)
    {
      double sum = 0.0;
      for (int k = 0; k < nx; k++)
      {
        sum += normal[y * nx + k] * trans[k * ny + x];
      }
      result[y * ny + x] = sum;
    }
  }
}
