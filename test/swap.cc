#include <cmath>
#include <vector>
#include <iostream>
#include <immintrin.h>

using namespace std;

typedef double double8_t __attribute__((vector_size(8 * sizeof(double))));

static inline double8_t swap1(double8_t x) { return _mm512_permute_pd(x, 0b01010101); }
static inline double8_t swap2(double8_t x)
{
  return double8_t{x[2], x[3], x[0], x[1], x[6], x[7], x[4], x[5]};
}
static inline double8_t swap4(double8_t x)
{
  return double8_t{x[4], x[5], x[6], x[7], x[0], x[1], x[2], x[3]};
}

void pr(double8_t v)
{
  cout << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << " " << v[4] << " " << v[5] << " " << v[6] << " " << v[7] << endl;
}

int main()
{
  double8_t a0 = double8_t{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  double8_t b0 = double8_t{9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

  pr(a0);
  double8_t a1 = swap1(a0);
  pr(a1);

  pr(b0);
  double8_t b1 = swap4(b0);
  pr(b1);
  double8_t b2 = swap2(b1);
  pr(b2);
  double8_t b3 = swap2(b0);
  pr(b2);

  double8_t a0b0 = a0 * b0;
  double8_t a0b1 = a0 * b1;
  double8_t a0b2 = a0 * b2;
  double8_t a0b3 = a0 * b3;

  double8_t a1b0 = a1 * b0;
  double8_t a1b1 = a1 * b1;
  double8_t a1b2 = a1 * b2;
  double8_t a1b3 = a1 * b3;

  vector<double> sums(64);

  sums[0] += a0b0[0];
  sums[1] += a1b0[1];
  sums[2] += a0b3[0];
  sums[3] += a1b3[1];
  sums[4] += a0b1[0];
  sums[5] += a1b1[1];
  sums[6] += a0b2[0];
  sums[7] += a1b2[1];

  sums[8] += a1b0[0];
  sums[9] += a0b0[1];
  sums[10] += a1b3[0];
  sums[11] += a0b3[1];
  sums[12] += a1b1[0];
  sums[13] += a0b1[1];
  sums[14] += a1b2[0];
  sums[15] += a0b2[1];

  sums[16] += a0b3[2];
  sums[17] += a1b3[3];
  sums[18] += a0b0[2];
  sums[19] += a1b0[3];
  sums[20] += a0b2[2];
  sums[21] += a1b2[3];
  sums[22] += a0b1[2];
  sums[23] += a1b1[3];

  sums[24] += a1b3[2];
  sums[25] += a0b3[3];
  sums[26] += a1b0[2];
  sums[27] += a0b0[3];
  sums[28] += a1b2[2];
  sums[29] += a0b2[3];
  sums[30] += a1b1[2];
  sums[31] += a0b1[3];

  sums[32] += a0b1[4];
  sums[33] += a1b1[5];
  sums[34] += a0b2[4];
  sums[35] += a1b2[5];
  sums[36] += a0b0[4];
  sums[37] += a1b0[5];
  sums[38] += a0b3[4];
  sums[39] += a1b3[5];

  sums[40] += a1b1[4];
  sums[41] += a0b1[5];
  sums[42] += a1b2[4];
  sums[43] += a0b2[5];
  sums[44] += a1b0[4];
  sums[45] += a0b0[5];
  sums[46] += a1b3[4];
  sums[47] += a0b3[5];

  sums[48] += a0b2[6];
  sums[49] += a1b2[7];
  sums[50] += a0b1[6];
  sums[51] += a1b1[7];
  sums[52] += a0b3[6];
  sums[53] += a1b3[7];
  sums[54] += a0b0[6];
  sums[55] += a1b0[7];

  sums[56] += a1b2[6];
  sums[57] += a0b2[7];
  sums[58] += a1b1[6];
  sums[59] += a0b1[7];
  sums[60] += a1b3[6];
  sums[61] += a0b3[7];
  sums[62] += a1b0[6];
  sums[63] += a0b0[7];

  cout << endl;
  cout << endl;

  for (int n = 0; n < 8; n++)
  {
    for (int k = 0; k < 8; k++)
    {
      cout << sums[n * 8 + k] << " ";
    }
    cout << endl;
  }
}