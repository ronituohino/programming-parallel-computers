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
  double4_t v1 = double4_t{1.0, 2.0, 3.0, 4.0};
  double4_t v2 = double4_t{5.0, 6.0, 7.0, 8.0};

  pr(v1);
  double4_t v1_1 = swap1(v1);
  pr(v1_1);

  double4_t v1_2 = swap2(v1);
  pr(v1_2);
  pr(v2);
  double4_t v2_1 = swap1(v2);
  pr(v2_1);
  double4_t v2_2 = swap2(v2);
  pr(v2_2);

  double4_t vv00 = v1 + v2;
  double4_t vv11 = v1_1 + v2_1;
  double4_t vv10 = v1_1 + v2;
  double4_t vv01 = v1 + v2_1;
  cout << endl;
  pr(vv00);
  pr(vv11);
  pr(vv10);
  pr(vv01);

  double res[28] = {1.0, 2.0};
  cout << res[0];
}