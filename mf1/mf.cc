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
  for (int y = 0; y < ny; y++)
  {
    for (int x = 0; x < nx; x++)
    {
      vector<float> pixels;

      // Scan the window
      for (int b = max(y - hy, 0); b < min(y + hy + 1, ny); b++)
      {
        for (int a = max(x - hx, 0); a < min(x + hx + 1, nx); a++)
        {
          pixels.push_back(in[b * nx + a]);
        }
      }

      if (pixels.size() % 2 == 0)
      {
        // Even amount of neighbors, need to calculate mean
        int mid1 = pixels.size() / 2 - 1;
        nth_element(pixels.begin(), pixels.begin() + mid1, pixels.end());
        float m1 = pixels[mid1];

        int mid2 = pixels.size() / 2;
        nth_element(pixels.begin(), pixels.begin() + mid2, pixels.end());
        float m2 = pixels[mid2];

        out[y * nx + x] = (m1 + m2) / 2.0;
      }
      else
      {
        // Odd amount of neighbors, median always in the middle
        int mid = pixels.size() / 2;
        nth_element(pixels.begin(), pixels.begin() + mid, pixels.end());
        out[y * nx + x] = pixels[mid];
      }
    }
  }
}
