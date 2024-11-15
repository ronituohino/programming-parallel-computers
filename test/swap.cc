#include <cmath>
#include <vector>
#include <iostream>
#include <immintrin.h>

using namespace std;

typedef double double8_t __attribute__((vector_size(8 * sizeof(double))));

static inline double8_t swap1(double8_t x) { return _mm512_permute_pd(x, 0b01010101); }
static inline double8_t swap2(double8_t x)
{
  return _mm512_shuffle_f64x2(x, x, 0b01010101);
}
static inline double8_t swap4(double8_t x)
{
  return _mm512_shuffle_f64x2(_mm512_unpacklo_pd(x, x), _mm512_unpackhi_pd(x, x), 0b01000100);
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
}