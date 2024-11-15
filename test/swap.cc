#include <cmath>
#include <vector>
#include <iostream>
#include <x86intrin.h>

#include <tuple>
#include <algorithm>

using namespace std;

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

static inline double4_t swap1(double4_t x) { return _mm256_permute_pd(x, 0b0101); }
static inline double4_t swap2(double4_t x)
{
  double4_t p = _mm256_permute_pd(x, 0b1010);
  return _mm256_permute2f128_pd(p, p, 0b00000001);
}

void pr(double4_t v)
{
  cout << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << endl;
}

int main()
{
  double4_t a0 = double4_t{1.0, 2.0, 3.0, 4.0};
  double4_t b0 = double4_t{5.0, 6.0, 7.0, 8.0};

  pr(a0);
  double4_t a1 = swap1(a0);
  pr(a1);
  double4_t a2 = swap2(a0);
  pr(a2);

  pr(b0);
  double4_t b1 = swap1(b0);
  pr(b1);
  double4_t b2 = swap2(b0);
  pr(b2);

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

  cout << endl;

  double sums[16];

  // sums[0] += a01[0];
  // sums[1] += a02[0];
  // sums[2] += a12[2];
  sums[0] += ab00[0];
  sums[1] += ab01[0];
  sums[2] += ab20[2];
  sums[3] += ab12[1];
  // sums[7] += a12[0];
  // sums[8] += a02[1];
  sums[4] += ab01[1];
  sums[5] += ab00[1];
  sums[6] += ab12[0];
  sums[7] += ab20[3];
  // sums[13] += a01[2];
  sums[8] += ab20[0];
  sums[9] += ab12[3];
  sums[10] += ab00[2];
  sums[11] += ab01[2];
  sums[12] += ab12[2];
  sums[13] += ab20[1];
  sums[14] += ab01[3];
  sums[15] += ab00[3];
  /*
  sums[22] += b01[0];
  sums[23] += b02[0];
  sums[24] += b12[1];
  sums[25] += b12[0];
  sums[26] += b02[1];
  sums[27] += b01[2];
  */

  for (int i = 0; i < 16; i++)
  {
    cout << sums[i] << endl;
  }

  cout << endl
       << endl;

  for (int k = 0; k < 7; k++)
  {
    for (int l = k + 1; l < 8; l++)
    {
      int n = (k * (15 - k)) / 2 + (l - k - 1);
      cout << n << endl;
    }
    cout << endl;
  }

  for (int n = 16; n < 22; n++)
  {
    int nx = (n - 16) / 3 + (n == 21 ? 1 : 0); // 0, 0, 0, 1, 1, 2
    int ns = (n < 19 ? n - 16 : (n == 19) ? 1
                                          : 2); // 0, 1, 2, 1, 2, 2
    cout << ns << endl;
  }

  int y_parts = 16;
  vector<tuple<int, int, int>> rows(y_parts * y_parts);
  for (int ia = 0; ia < y_parts; ++ia)
  {
    for (int ja = 0; ja < y_parts; ++ja)
    {
      int ija = _pdep_u32(ia, 0x55555555) | _pdep_u32(ja, 0xAAAAAAAA);
      rows[ia * y_parts + ja] = make_tuple(ija, ia, ja);
    }
  }
  sort(rows.begin(), rows.end());
  for (int n = 0; n < 16; n++)
  {
    auto [a, b, c] = rows[n];
    cout << a << " " << b << " " << c << endl;
  }
}