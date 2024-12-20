struct Result
{
    float avg[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- horizontal position: 0 <= x0 < x1 <= nx
- vertical position: 0 <= y0 < y1 <= ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
- output: avg[c]
*/
Result calculate(int ny, int nx, const float *data, int y0, int x0, int y1, int x1)
{
    double total_c[3] = {0.0, 0.0, 0.0};
    for (int y = y0; y < y1; y++)
    {
        for (int x = x0; x < x1; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                total_c[c] += data[c + 3 * x + 3 * nx * y];
            }
        }
    }
    const int data_points = (y1 - y0) * (x1 - x0);
    total_c[0] = total_c[0] / data_points;
    total_c[1] = total_c[1] / data_points;
    total_c[2] = total_c[2] / data_points;

    return Result{{(float)total_c[0], (float)total_c[1], (float)total_c[2]}};
    ;
}
