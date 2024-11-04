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

struct Sum
{
    double r;
    double g;
    double b;
};
Sum get_rect_sum(vector<double> const &table, int nx, int x0, int x1, int y0, int y1)
{
    double sums[3];
    for (int c = 0; c < 3; c++)
    {
        double sum_to_rect_br = table[c + 3 * (x1 - 1) + 3 * nx * (y1 - 1)];
        double sum_to_rect_tl = (x0 > 0 && y0 > 0) ? table[c + 3 * (x0 - 1) + 3 * nx * (y0 - 1)] : 0.0;
        double sum_to_rect_tr = (y0 > 0) ? table[c + 3 * (x1 - 1) + 3 * nx * (y0 - 1)] : 0.0;
        double sum_to_rect_bl = (x0 > 0) ? table[c + 3 * (x0 - 1) + 3 * nx * (y1 - 1)] : 0.0;
        sums[c] = sum_to_rect_br - sum_to_rect_tr - sum_to_rect_bl + sum_to_rect_tl;
    }
    return Sum{sums[0], sums[1], sums[2]};
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
    vector<double> sums_of_pixels(ny * nx * 3);
    vector<double> sq_sums_of_pixels(ny * nx * 3);
    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx; x++)
        {
            double total[3];
            for (int c = 0; c < 3; c++)
            {
                total[c] = data[c + 3 * x + 3 * nx * y];
                if (x > 0)
                {
                    total[c] += sums_of_pixels[c + 3 * (x - 1) + 3 * nx * y];
                }
                if (y > 0)
                {
                    total[c] += sums_of_pixels[c + 3 * x + 3 * nx * (y - 1)];
                }
                if (x > 0 && y > 0)
                {
                    total[c] -= sums_of_pixels[c + 3 * (x - 1) + 3 * nx * (y - 1)];
                }
                sums_of_pixels[c + 3 * x + 3 * nx * y] = total[c];
                sq_sums_of_pixels[c + 3 * x + 3 * nx * y] = pow(total[c], 2);
            }
        }
    }

    // Get sum and sq_sum for entire image
    double sum_for_img[3];
    double sq_sum_for_img[3];
    for (int c = 0; c < 3; c++)
    {
        sum_for_img[c] = sums_of_pixels[ny * nx * 3 - (3 - c)];
        sq_sum_for_img[c] = sq_sums_of_pixels[ny * nx * 3 - (3 - c)];
    }
    int img_dim = nx * ny;

    Result min{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    double min_cost = numeric_limits<double>::infinity();

    // Move a sliding window over the original image starting from top-left
    // going down and then to the top and a bit to the right
    // After full scan of the original image, make the window smaller.
    for (int h = ny; h > 0; h--)
    {
        for (int w = nx; w > 0; w--)
        {
            // Ignore case where window is size of image
            if (w == nx && h == ny)
            {
                continue;
            }
            int rect_dim = w * h;
            int out_dim = img_dim - rect_dim;
            for (int y0 = 0; y0 <= ny - h; y0++)
            {
                for (int x0 = 0; x0 <= nx - w; x0++)
                {
                    int x1 = x0 + w;
                    int y1 = y0 + h;

                    Sum s1 = get_rect_sum(sums_of_pixels, nx, x0, x1, y0, y1);
                    double sum_for_rect[3] = {s1.r, s1.g, s1.b};
                    Sum s2 = get_rect_sum(sq_sums_of_pixels, nx, x0, x1, y0, y1);
                    double sq_sum_for_rect[3] = {s2.r, s2.g, s2.b};

                    double sum_for_out[3];
                    double error1 = 0.0;
                    double error2 = 0.0;

                    for (int c = 0; c < 3; c++)
                    {
                        double av_rect = sum_for_rect[c] / rect_dim;
                        sum_for_out[c] = sum_for_img[c] - sum_for_rect[c];
                        double av_out = sum_for_out[c] / out_dim;
                        double sq_sum_for_out = sq_sum_for_img[c] - sq_sum_for_rect[c];
                        error1 += sq_sum_for_rect[c] - 2 * av_rect * sum_for_rect[c] + rect_dim * pow(av_rect, 2);
                        error2 += sq_sum_for_out - 2 * av_out * sum_for_out[c] + out_dim * pow(av_out, 2);
                    }

                    double cost = error1 + error2;

                    if (cost < min_cost)
                    {
                        min_cost = cost;
                        min = {
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
    }
    return min;
}