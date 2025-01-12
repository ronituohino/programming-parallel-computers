#include <algorithm>
#include <iostream>
#include <vector>

typedef unsigned long long data_t;

using namespace std;

int bin_search(data_t value, data_t *data, int start, int end)
{
    int low = start;
    int high = max(start, end + 1);
    while (low < high)
    {
        int mid = (low + high) / 2;
        if (value <= data[mid])
        {
            high = mid;
        }
        else
        {
            low = mid + 1;
        }
    }
    return high;
}

void p_merge(data_t *data, int s1, int e1, int s2, int e2, data_t *buf, int s3)
{
    int l1 = e1 - s1;
    int l2 = e2 - s2;

    // Make sure left is larger or equal size to right
    if (l1 < l2)
    {
        int s3 = s1;
        s1 = s2;
        s2 = s3;

        int e3 = e1;
        e1 = e2;
        e2 = e3;

        int l3 = l1;
        l1 = l2;
        l2 = l3;
    }

    if (l1 == 0)
    {
        return;
    }
    else
    {
        int m1 = s1 + l1 / 2;
        int m2 = bin_search(data[m1], data, s2, e2);
        int m3 = s3 + (m1 - s1) + (m2 - s2);
        buf[m3] = data[m1];

        p_merge(data, s1, m1 - 1, s2, m2 - 1, buf, s3);
        p_merge(data, m1 + 1, e1, m2, e2, buf, m3 + 1);
    }
}

void pm_sort(data_t *from, data_t *to, int s, int e, int thr, data_t *data)
{
    int l = e - s;
    if (l < thr)
    {
        // For small subsets, use serial sort
        if (data != to)
        {
            // copy
            for (int i = s; i < e; i++)
            {
                to[i] = data[i];
            }
        }
        sort(to, to + l + 1);
    }
    else
    {
        int m = s + l / 2;
        pm_sort(to, from, s, m, thr, data);
        pm_sort(to, from, m + 1, e, thr, data);

        p_merge(from, s, m, m + 1, e, to, s);
    }
}

void psort(int n, data_t *data)
{
    data_t *aux = (data_t *)malloc(n * sizeof(data_t));
    int thresh = 10000000;
    pm_sort(aux, data, 0, n - 1, thresh, data);
    free(aux);
}