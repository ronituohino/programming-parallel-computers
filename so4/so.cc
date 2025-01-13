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

void s_merge(data_t *data, int s1, int e1, int s2, int e2, data_t *buf, int s3)
{
    int l1 = e1 - s1;
    int l2 = e2 - s2;

    int i = 0, j = 0, k = s3;
    while (i <= l1 && j <= l2)
    {
        if (data[s1 + i] <= data[s2 + j])
        {
            // left
            buf[k] = data[s1 + i];
            i++;
        }
        else
        {
            // right
            buf[k] = data[s2 + j];
            j++;
        }
        k++;
    }

    while (i <= l1)
    {
        // left
        buf[k] = data[s1 + i];
        i++;
        k++;
    }

    while (j <= l2)
    {
        // right
        buf[k] = data[s2 + j];
        j++;
        k++;
    }
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

    if (l1 < 10000000)
    {
        s_merge(data, s1, e1, s2, e2, buf, s3);
    }
    else
    {
        int m1 = s1 + l1 / 2;
        int m2 = bin_search(data[m1], data, s2, e2);
        int m3 = s3 + (m1 - s1) + (m2 - s2);

        buf[m3] = data[m1];

#pragma omp parallel sections
        {
#pragma omp section
            p_merge(data, s1, m1 - 1, s2, m2 - 1, buf, s3);
#pragma omp section
            p_merge(data, m1 + 1, e1, m2, e2, buf, m3 + 1);
        }
    }
}

void pm_sort(data_t *from, data_t *to, int s, int e, data_t *data)
{
    int l = e - s;
    if (l < 10000000)
    {
        // For small subsets, use serial sort
        if (data != to)
        {
            // copy
            for (int i = s; i <= e; i++)
            {
                to[i] = data[i];
            }
        }
        sort(to + s, to + e + 1);
    }
    else
    {
        int m = s + l / 2;

#pragma omp parallel sections
        {
#pragma omp section
            pm_sort(to, from, s, m, data);
#pragma omp section
            pm_sort(to, from, m + 1, e, data);
        }

        p_merge(from, s, m, m + 1, e, to, s);
    }
}

void psort(int n, data_t *data)
{
    data_t *aux = (data_t *)malloc(n * sizeof(data_t));
    pm_sort(aux, data, 0, n - 1, data);
    free(aux);
}