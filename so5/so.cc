#include <algorithm>
#include <iostream>

typedef unsigned long long data_t;

using namespace std;

struct Partition
{
    int i;
    int j;
};

Partition partition(data_t *d, int lo, int hi, int i, int j)
{
    // Median of three pivot selection
    int mid = (lo + hi) / 2;
    if (d[mid] < d[lo])
    {
        swap(d[lo], d[mid]);
    }
    if (d[hi] < d[lo])
    {
        swap(d[lo], d[hi]);
    }
    if (d[mid] < d[hi])
    {
        swap(d[mid], d[hi]);
    }
    data_t pivot = d[hi];

    while (i <= j)
    {
        while (d[i] < pivot)
        {
            i += 1;
        }
        while (d[j] > pivot)
        {
            j -= 1;
        }
        if (i <= j)
        {
            swap(d[i], d[j]);
            i += 1;
            j -= 1;
        }
    }
    return Partition{i, j};
}

void quicksort(data_t *data, int lo, int hi)
{
    if (lo < 0 || hi < 0 || hi <= lo)
    {
        return;
    }
    else if (hi - lo <= 100000)
    {
        sort(data + lo, data + hi + 1);
    }
    else
    {
        Partition p = partition(data, lo, hi, lo, hi);
#pragma omp task
        quicksort(data, lo, p.j);
#pragma omp task
        quicksort(data, p.i, hi);
    }
}

void psort(int n, data_t *d)
{
#pragma omp parallel
#pragma omp single
    {
        quicksort(d, 0, n - 1);
    }
}