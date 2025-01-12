#include <algorithm>
#include <iostream>
#include <vector>

typedef unsigned long long data_t;

using namespace std;

void merge(data_t *data, int left, int mid, int right)
{
    int n1 = mid - left + 1;
    int n2 = right - mid;

    data_t *leftBuff = (data_t *)malloc(n1 * sizeof(data_t));
    data_t *rightBuff = (data_t *)malloc(n2 * sizeof(data_t));

    for (int i = 0; i < n1; ++i)
    {
        leftBuff[i] = data[left + i];
    }
    for (int i = 0; i < n2; ++i)
    {
        rightBuff[i] = data[mid + 1 + i];
    }

    // Merge the buffers back into the original data
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
    {
        if (leftBuff[i] <= rightBuff[j])
        {
            data[k] = leftBuff[i];
            i++;
        }
        else
        {
            data[k] = rightBuff[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of leftBuff
    while (i < n1)
    {
        data[k] = leftBuff[i];
        i++;
        k++;
    }

    // Copy the remaining elements of rightBuff
    while (j < n2)
    {
        data[k] = rightBuff[j];
        j++;
        k++;
    }

    free(leftBuff);
    free(rightBuff);
}

void pms(data_t *data, int left, int right, int thresh)
{
    // Subsets with only 1 element are already "sorted"
    if (left < right)
    {
        int len = right - left;
        int mid = left + len / 2;

        if (len < thresh)
        {
            // For small subsets, use serial sort
            sort(data + left, data + right + 1);
        }
        else
        {
#pragma omp parallel sections
            {
#pragma omp section
                pms(data, left, mid, thresh);

#pragma omp section
                pms(data, mid + 1, right, thresh);
            }
        }

        merge(data, left, mid, right);
    }
}

void psort(int n, data_t *data)
{
    int thresh = 100000;
    pms(data, 0, n - 1, thresh);
}