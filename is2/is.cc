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
    double sum_to_rect_br[3] = {
        table[0 + 3 * (x1 - 1) + 3 * nx * (y1 - 1)],
        table[1 + 3 * (x1 - 1) + 3 * nx * (y1 - 1)],
        table[2 + 3 * (x1 - 1) + 3 * nx * (y1 - 1)]};

    double sum_to_rect_tl[3] = {
        (x0 > 0 && y0 > 0) ? table[0 + 3 * (x0 - 1) + 3 * nx * (y0 - 1)] : 0.0,
        (x0 > 0 && y0 > 0) ? table[1 + 3 * (x0 - 1) + 3 * nx * (y0 - 1)] : 0.0,
        (x0 > 0 && y0 > 0) ? table[2 + 3 * (x0 - 1) + 3 * nx * (y0 - 1)] : 0.0};

    double sum_to_rect_tr[3] = {
        y0 > 0 ? table[0 + 3 * (x1 - 1) + 3 * nx * (y0 - 1)] : 0.0,
        y0 > 0 ? table[1 + 3 * (x1 - 1) + 3 * nx * (y0 - 1)] : 0.0,
        y0 > 0 ? table[2 + 3 * (x1 - 1) + 3 * nx * (y0 - 1)] : 0.0};

    double sum_to_rect_bl[3] = {
        x0 > 0 ? table[0 + 3 * (x0 - 1) + 3 * nx * (y1 - 1)] : 0.0,
        x0 > 0 ? table[1 + 3 * (x0 - 1) + 3 * nx * (y1 - 1)] : 0.0,
        x0 > 0 ? table[2 + 3 * (x0 - 1) + 3 * nx * (y1 - 1)] : 0.0};

    Sum sum_for_rect = {
        sum_to_rect_br[0] -
            sum_to_rect_tr[0] -
            sum_to_rect_bl[0] +
            sum_to_rect_tl[0],

        sum_to_rect_br[1] -
            sum_to_rect_tr[1] -
            sum_to_rect_bl[1] +
            sum_to_rect_tl[1],

        sum_to_rect_br[2] -
            sum_to_rect_tr[2] -
            sum_to_rect_bl[2] +
            sum_to_rect_tl[2]};
    return sum_for_rect;
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
            double total_r = data[0 + 3 * x + 3 * nx * y];
            double total_g = data[1 + 3 * x + 3 * nx * y];
            double total_b = data[2 + 3 * x + 3 * nx * y];
            if (x > 0)
            {
                total_r += sums_of_pixels[0 + 3 * (x - 1) + 3 * nx * y];
                total_g += sums_of_pixels[1 + 3 * (x - 1) + 3 * nx * y];
                total_b += sums_of_pixels[2 + 3 * (x - 1) + 3 * nx * y];
            }
            if (y > 0)
            {
                total_r += sums_of_pixels[0 + 3 * x + 3 * nx * (y - 1)];
                total_g += sums_of_pixels[1 + 3 * x + 3 * nx * (y - 1)];
                total_b += sums_of_pixels[2 + 3 * x + 3 * nx * (y - 1)];
            }
            if (x > 0 && y > 0)
            {
                total_r -= sums_of_pixels[0 + 3 * (x - 1) + 3 * nx * (y - 1)];
                total_g -= sums_of_pixels[1 + 3 * (x - 1) + 3 * nx * (y - 1)];
                total_b -= sums_of_pixels[2 + 3 * (x - 1) + 3 * nx * (y - 1)];
            }

            sums_of_pixels[0 + 3 * x + 3 * nx * y] = total_r;
            sums_of_pixels[1 + 3 * x + 3 * nx * y] = total_g;
            sums_of_pixels[2 + 3 * x + 3 * nx * y] = total_b;

            sq_sums_of_pixels[0 + 3 * x + 3 * nx * y] = pow(total_r, 2);
            sq_sums_of_pixels[1 + 3 * x + 3 * nx * y] = pow(total_g, 2);
            sq_sums_of_pixels[2 + 3 * x + 3 * nx * y] = pow(total_b, 2);
        }
    }

    // Move a sliding window over the original image starting from top-left
    // going down and then to the top and a bit to the right
    // After full scan of the original image, make the window smaller.

    Sum sum_for_img = {
        sums_of_pixels[ny * nx * 3 - 3],
        sums_of_pixels[ny * nx * 3 - 2],
        sums_of_pixels[ny * nx * 3 - 1]};

    Sum sq_sum_for_img = {
        sq_sums_of_pixels[ny * nx * 3 - 3],
        sq_sums_of_pixels[ny * nx * 3 - 2],
        sq_sums_of_pixels[ny * nx * 3 - 1]};

    int img_dim = nx * ny;

    Result min{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    double min_cost = numeric_limits<double>::infinity();

    for (int h = ny; h > 0; h--)
    {
        for (int w = nx; w > 0; w--)
        {
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

                    Sum sum_for_rect = get_rect_sum(sums_of_pixels, nx, x0, x1, y0, y1);
                    Sum sq_sum_for_rect = get_rect_sum(sq_sums_of_pixels, nx, x0, x1, y0, y1);

                    double error1 = (sq_sum_for_rect.r - 2 * (sum_for_rect.r / rect_dim) * sum_for_rect.r + rect_dim * pow(sum_for_rect.r / rect_dim, 2) +
                                     sq_sum_for_rect.g - 2 * (sum_for_rect.g / rect_dim) * sum_for_rect.g + rect_dim * pow(sum_for_rect.g / rect_dim, 2) +
                                     sq_sum_for_rect.b - 2 * (sum_for_rect.b / rect_dim) * sum_for_rect.b + rect_dim * pow(sum_for_rect.b / rect_dim, 2));

                    Sum sum_for_out = {
                        (sum_for_img.r - sum_for_rect.r),
                        (sum_for_img.g - sum_for_rect.g),
                        (sum_for_img.b - sum_for_rect.b)};
                    Sum sq_sum_for_out = {
                        (sq_sum_for_img.r - sq_sum_for_rect.r),
                        (sq_sum_for_img.g - sq_sum_for_rect.g),
                        (sq_sum_for_img.b - sq_sum_for_rect.b)};

                    double error2 = (sq_sum_for_out.r - 2 * (sum_for_out.r / out_dim) * sum_for_out.r + out_dim * pow(sum_for_out.r / out_dim, 2) +
                                     sq_sum_for_out.g - 2 * (sum_for_out.g / out_dim) * sum_for_out.g + out_dim * pow(sum_for_out.g / out_dim, 2) +
                                     sq_sum_for_out.b - 2 * (sum_for_out.b / out_dim) * sum_for_out.b + out_dim * pow(sum_for_out.b / out_dim, 2));

                    double cost = error1 + error2;

                    if (cost > 0 && cost < min_cost)
                    {
                        min_cost = cost;
                        min = {
                            y0,
                            x0,
                            y1,
                            x1,
                            {(float)(sum_for_out.r / out_dim),
                             (float)(sum_for_out.g / out_dim),
                             (float)(sum_for_out.b / out_dim)},
                            {(float)(sum_for_rect.r / rect_dim),
                             (float)(sum_for_rect.g / rect_dim),
                             (float)(sum_for_rect.b / rect_dim)}};
                    }
                }
            }
        }
    }
    return min;
}