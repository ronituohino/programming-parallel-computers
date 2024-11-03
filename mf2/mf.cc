/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
#include <vector>
#include <algorithm>

using namespace std;

void mf(int ny, int nx, int hy, int hx, const float *in, float *out)
{
// Scan the grid
#pragma omp parallel for
  for (int y = 0; y < ny; y++)
  {
    for (int x = 0; x < nx; x++)
    {
      int b_min = max(y - hy, 0);
      int b_max = min(y + hy + 1, ny);
      int w_height = (b_max - b_min);
      int a_min = max(x - hx, 0);
      int a_max = min(x + hx + 1, nx);
      int w_width = (a_max - a_min);

      int pixels_size = w_height * w_width;
      vector<float> pixels(pixels_size, 0.0);

      // Scan the window
      for (int b = b_min; b < b_max; b++)
      {
        for (int a = a_min; a < a_max; a++)
        {
          pixels[(b - b_min) * w_width + (a - a_min)] = in[b * nx + a];
        }
      }

      int mid = pixels_size / 2;
      nth_element(pixels.begin(), pixels.begin() + mid, pixels.end());
      float m = pixels[mid];

      if (pixels_size % 2 == 0)
      {
        // Even amount of neighbors, need to calculate mean
        int mid2 = pixels_size / 2 - 1;
        nth_element(pixels.begin(), pixels.begin() + mid2, pixels.end());
        float m2 = pixels[mid2];

        out[y * nx + x] = (m + m2) / 2.0;
      }
      else
      {
        // Odd amount of neighbors, median always in the middle
        out[y * nx + x] = m;
      }
    }
  }
}
