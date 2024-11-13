#include <cmath>
#include <vector>
#include <iostream>
#include <x86intrin.h>

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

  double res[28] = {
      a01[0],
      a02[0],
      a12[2],
      ab00[0],
      ab01[0],
      ab20[2],
      ab12[1],
      a12[0],
      a02[1],
      ab01[1],
      ab00[1],
      ab12[0],
      ab20[3],
      a01[2],
      ab20[0],
      ab12[3],
      ab00[2],
      ab01[2],
      ab12[2],
      ab20[1],
      ab01[3],
      ab00[3],
      b01[0],
      b02[0],
      b12[1],
      b12[0],
      b02[1],
      b01[2],
  };

  for (int i = 0; i < 28; i++)
  {
    cout << res[i] << endl;
  }
}