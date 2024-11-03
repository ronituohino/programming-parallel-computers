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
        table[0 + 3 * x1 + 3 * nx * y1],
        table[1 + 3 * x1 + 3 * nx * y1],
        table[2 + 3 * x1 + 3 * nx * y1]};

    double sum_to_rect_tl[3] = {
        x0 > 0 ? table[0 + 3 * (x0 - 1) + 3 * nx * y0] : 0.0,
        x0 > 0 ? table[1 + 3 * (x0 - 1) + 3 * nx * y0] : 0.0,
        x0 > 0 ? table[2 + 3 * (x0 - 1) + 3 * nx * y0] : 0.0};

    double sum_to_rect_tr[3] = {
        y0 > 0 ? table[0 + 3 * x1 + 3 * nx * (y0 - 1)] : 0.0,
        y0 > 0 ? table[1 + 3 * x1 + 3 * nx * (y0 - 1)] : 0.0,
        y0 > 0 ? table[2 + 3 * x1 + 3 * nx * (y0 - 1)] : 0.0};

    double sum_to_rect_bl[3] = {
        x0 > 0 ? table[0 + 3 * (x0 - 1) + 3 * nx * y1] : 0.0,
        x0 > 0 ? table[1 + 3 * (x0 - 1) + 3 * nx * y1] : 0.0,
        x0 > 0 ? table[2 + 3 * (x0 - 1) + 3 * nx * y1] : 0.0};

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
        sums_of_pixels[ny * (nx - 1) * 3 + 0],
        sums_of_pixels[ny * (nx - 1) * 3 + 1],
        sums_of_pixels[ny * (nx - 1) * 3 + 2]};

    Sum sq_sum_for_img = {
        sq_sums_of_pixels[ny * (nx - 1) * 3 + 0],
        sq_sums_of_pixels[ny * (nx - 1) * 3 + 1],
        sq_sums_of_pixels[ny * (nx - 1) * 3 + 2]};

    int img_dim = nx * ny;

    Result min{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    double min_cost = numeric_limits<double>::infinity();

    for (int h = ny; h > 0; h--)
    {
        for (int w = nx; w > 0; w--)
        {
            int rect_dim = w * h;
            for (int y0 = 0; y0 <= ny - h; y0++)
            {
                for (int x0 = 0; x0 <= nx - w; x0++)
                {
                    int x1 = x0 + w - 1;
                    int y1 = y0 + h - 1;

                    cout << "Handling this: ";
                    cout << x0 << x1 << y0 << y1 << endl;
                    cout << w << h;
                    cout << endl;

                    Sum sum_for_rect = get_rect_sum(sums_of_pixels, nx, x0, x1, y0, y1);
                    Sum sq_sum_for_rect = get_rect_sum(sq_sums_of_pixels, nx, x0, x1, y0, y1);

                    double inner_avg_col[3] = {
                        sum_for_rect.r / rect_dim,
                        sum_for_rect.g / rect_dim,
                        sum_for_rect.b / rect_dim};

                    double outer_avg_col[3] = {
                        (sum_for_img.r - sum_for_rect.r) / img_dim,
                        (sum_for_img.g - sum_for_rect.g) / img_dim,
                        (sum_for_img.b - sum_for_rect.b) / img_dim};

                    double error1 = (sq_sum_for_rect.r - 2 * inner_avg_col[0] * sum_for_rect.r + rect_dim * pow(inner_avg_col[0], 2) +
                                     sq_sum_for_rect.g - 2 * inner_avg_col[1] * sum_for_rect.g + rect_dim * pow(inner_avg_col[1], 2) +
                                     sq_sum_for_rect.b - 2 * inner_avg_col[2] * sum_for_rect.b + rect_dim * pow(inner_avg_col[2], 2));

                    double error2 = (sq_sum_for_img.r - 2 * outer_avg_col[0] * sum_for_img.r + img_dim * pow(outer_avg_col[0], 2) +
                                     sq_sum_for_img.g - 2 * outer_avg_col[1] * sum_for_img.g + img_dim * pow(outer_avg_col[1], 2) +
                                     sq_sum_for_img.b - 2 * outer_avg_col[2] * sum_for_img.b + img_dim * pow(outer_avg_col[2], 2));

                    double cost = error1 + error2;

                    cout << "Cost: ";
                    cout << cost << endl;

                    if (cost < min_cost)
                    {
                        min = {
                            y0,
                            x0,
                            y1,
                            x1,
                            {(float)outer_avg_col[0], (float)outer_avg_col[1], (float)outer_avg_col[2]},
                            {(float)inner_avg_col[0], (float)inner_avg_col[1], (float)inner_avg_col[2]}};
                    }
                }
            }
        }
    }
    return min;
}

int main()
{
    float data[9] = {
        +0.31532657, +0.11138125, +0.19794576,
        +0.31532657, +0.11138125, +0.74090004,
        +0.31532657, +0.11138125, +0.19794576};

    segment(1, 3, data);
}