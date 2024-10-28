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
void correlate(int ny, int nx, const float *data, float *result)
{
  // Calculate median values for rows
  std::vector<double> median;
  median.assign(ny, 0.0);

  for (int y = 0; y < ny; y++)
  {
    double total = 0.0;
    for (int x = 0; x < nx; x++)
    {
      total += data[y * nx + x];
    }
    median[y] = total / nx;
  }

  // Calculate Pearson's correlation coefficient between all rows
  for (int j = 0; j < ny; j++)
  {
    for (int i = j; i < ny; i++)
    {
      // Top part of formula
      double top_sum = 0.0;
      double bottom_left_sum = 0.0;
      double bottom_right_sum = 0.0;
      for (int x = 0; x < nx; x++)
      {
        double x0 = data[i * nx + x] - median[i];
        double x1 = data[j * nx + x] - median[j];

        top_sum += x0 * x1;
        bottom_left_sum += pow(x0, 2);
        bottom_right_sum += pow(x1, 2);
      }

      double r = top_sum / (sqrt(bottom_left_sum) * sqrt(bottom_right_sum));
      result[i + j * ny] = (float)r;
    }
  }
}
