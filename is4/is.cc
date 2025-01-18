struct Result
{
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

constexpr double4_t empt = double4_t{0.0, 0.0, 0.0, 0.0};

double4_t get_rect_sum(vector<double4_t> const &table, int nx, int x0, int x1, int y0, int y1)
{
    double4_t sum_to_rect_br = table[(x1 - 1) + nx * (y1 - 1)];
    double4_t sum_to_rect_tl = (x0 > 0 && y0 > 0) ? table[(x0 - 1) + nx * (y0 - 1)] : empt;
    double4_t sum_to_rect_tr = (y0 > 0) ? table[(x1 - 1) + nx * (y0 - 1)] : empt;
    double4_t sum_to_rect_bl = (x0 > 0) ? table[(x0 - 1) + nx * (y1 - 1)] : empt;
    return sum_to_rect_br - sum_to_rect_tr - sum_to_rect_bl + sum_to_rect_tl;
}

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data)
{
    // Create matrix for looking up the sum of pixel
    // values from the top-left of the original image
    vector<double4_t> sums_of_pixels(ny * nx);
    vector<double4_t> sq_sums_of_pixels(ny * nx);
    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx; x++)
        {
            double4_t v;
            double4_t p;
            for (int c = 0; c < 3; c++)
            {
                double d = data[c + 3 * x + 3 * nx * y];
                v[c] = d;
                p[c] = pow(d, 2);
            }
            v[3] = 0.0;

            if (x > 0)
            {
                v += sums_of_pixels[(x - 1) + nx * y];
            }
            if (y > 0)
            {
                v += sums_of_pixels[x + nx * (y - 1)];
            }
            if (x > 0 && y > 0)
            {
                v -= sums_of_pixels[(x - 1) + nx * (y - 1)];
            }
            sums_of_pixels[x + nx * y] = v;
            sq_sums_of_pixels[x + nx * y] = p;
        }
    }

    // Get sum and sq_sum for entire image
    double4_t sum_for_img = sums_of_pixels[(ny - 1) * (nx - 1)];
    double4_t sq_sum_for_img = sq_sums_of_pixels[(ny - 1) * (nx - 1)];
    int img_dim = nx * ny;

    Result min;
    double min_cost = numeric_limits<double>::infinity();

// Move a sliding window over the original image starting from top-left
// going down and then to the top and a bit to the right
// After full scan of the original image, make the window smaller.
#pragma omp parallel for schedule(dynamic, 1)
    for (int h = ny; h > 0; h--)
    {
        Result local_min;
        double local_min_cost = numeric_limits<double>::infinity();

        for (int w = nx; w > 0; w--)
        {
            // Ignore case where window is size of image
            if (w == nx && h == ny)
            {
                continue;
            }
            double rect_dim = w * h;
            double4_t v_rect_dim = {rect_dim, rect_dim, rect_dim, rect_dim};
            double out_dim = img_dim - rect_dim;
            double4_t v_out_dim = {out_dim, out_dim, out_dim, out_dim};
            for (int y0 = 0; y0 <= ny - h; y0++)
            {
                for (int x0 = 0; x0 <= nx - w; x0++)
                {
                    int x1 = x0 + w;
                    int y1 = y0 + h;

                    double4_t sum_for_rect = get_rect_sum(sums_of_pixels, nx, x0, x1, y0, y1);
                    double4_t sq_sum_for_rect = get_rect_sum(sq_sums_of_pixels, nx, x0, x1, y0, y1);

                    double4_t sum_for_out = sum_for_img - sum_for_rect;

                    double4_t av_rect = sum_for_rect / v_rect_dim;
                    double4_t two_av_rect = {av_rect[0] * 2, av_rect[1] * 2, av_rect[2] * 2, av_rect[3] * 2};
                    double4_t pow_av_rect = {pow(av_rect[0], 2), pow(av_rect[1], 2), pow(av_rect[2], 2), 0.0};

                    double4_t av_out = sum_for_out / v_out_dim;
                    double4_t two_av_out = {av_out[0] * 2, av_out[1] * 2, av_out[2] * 2, av_out[3] * 2};
                    double4_t pow_av_out = {pow(av_out[0], 2), pow(av_out[1], 2), pow(av_out[2], 2), 0.0};

                    double4_t sq_sum_for_out = sq_sum_for_img - sq_sum_for_rect;

                    double4_t error1 = sq_sum_for_rect - two_av_rect * sum_for_rect + v_rect_dim * pow_av_rect;
                    double4_t error2 = sq_sum_for_out - two_av_out * sum_for_out + v_out_dim * pow_av_out;

                    double cost = 0.0;
                    for (int i = 0; i < 3; i++)
                    {
                        cost += error1[i];
                        cost += error2[i];
                    }

                    if (cost < local_min_cost)
                    {
                        local_min_cost = cost;
                        local_min = {
                            y0,
                            x0,
                            y1,
                            x1,
                            {(float)(sum_for_out[0] / out_dim),
                             (float)(sum_for_out[1] / out_dim),
                             (float)(sum_for_out[2] / out_dim)},
                            {(float)(sum_for_rect[0] / rect_dim),
                             (float)(sum_for_rect[1] / rect_dim),
                             (float)(sum_for_rect[2] / rect_dim)}};
                    }
                }
            }
        }

#pragma omp critical
        {
            if (local_min_cost < min_cost)
            {
                min_cost = local_min_cost;
                min = local_min;
            }
        }
    }
    return min;
}