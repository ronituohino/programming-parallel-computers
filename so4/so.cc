#include <algorithm>
#include <iostream>
#include <vector>

typedef unsigned long long data_t;

using namespace std;

void sort_in_vector(int len, data_t *data, vector<data_t> &output)
{
    for (int v = 0; v < len; v++)
    {
        data_t d = *data;
        output[v] = d;
    }
    sort(output.begin(), output.end());
}

void psort(int n, data_t *data)
{
    constexpr int parts = 2;
    int l = n / parts; // the approx length of each sub-array
    int approx_total = l * parts;
    int leftover = n - approx_total; // how many elements were left over

    vector<data_t> res1; // the first array has the extra elements (l + leftover)
    vector<data_t> res2; // the rest are the same size (l)

    cout << data << " " << endl;
    cout << *data << " " << endl;
    data++;
    cout << *data << " " << endl;

    // disassemble data into parts, and sort them in those parts
    for (int i = 0; i < parts; i++)
    {
        int start = (i * l) + leftover;
        int end = ((i + 1) * l) + leftover - 1;
        if (i == 0)
        {
            start = 0;
        }
        int len = end - start;

        // sort_in_vector(len, data, res1);
    }

    // assemble sorted parts back into data
    // cout << res1[0] << " " << res1[1] << endl;
    // cout << res2[0] << " " << res2[1] << endl;
}
