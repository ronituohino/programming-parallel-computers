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
  // Apply normalization
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

  vector<double> sums;
  for (int y = 0; y < ny; y++)
  {
    for (int x = 0; x < ny; x++)
    {
      sums.assign(slices, 0.0);
      for (int k = 0; k < nxp / slices; k++)
      {
        for (int s = 0; s < slices; s++)
        {
          sums[s] += padded[y * nxp + (k * slices) + s] * padded[x * nxp + (k * slices) + s];
        }
      }
      result[y * ny + x] = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    }
  }
}
