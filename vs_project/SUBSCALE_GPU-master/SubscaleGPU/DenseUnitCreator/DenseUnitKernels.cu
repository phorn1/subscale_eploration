#include "DenseUnitKernels.cuh"

// Binary Search for an integer that fulfills following conditions:
// 1. Smaller than or equal to end
// 2. Binomial coefficient "integer choose b" is smaller than or equal to target
// 3. As large as possible without violating condition 1 and 2
__device__ int binarySearch(const unsigned long long target, int end, const int b, BinomialCoeffCreator* binomCoeffs)
{
    int start = 0;

    int ans = -1;
    while (start <= end)
    {
        int mid = start + ((end - start) / 2);

        // Move to the left side if the target is smaller  
        if (binomCoeffs->choose(mid, b) > target)
        {
            end = mid - 1;
        }
        else // Move to the right side  
        {
            ans = mid;
            start = mid + 1;
        }
    }
    return ans;
}


// Naive Search for an integer that fulfills following conditions:
// 1. Smaller than or equal to end
// 2. Binomial coefficient "integer choose b" is smaller than or equal to target
// 3. As large as possible without violating condition 1 and 2
__device__ int naiveSearch(const unsigned long long target, int end, const int b, BinomialCoeffCreator* binomCoeffs)
{
    int ans = end;
    while (binomCoeffs->choose(ans, b) > target)
    {
        ans--;
    }

    return ans;
}


// Kernel for calculating a single dense unit (DEPRECATED)
__global__ void denseUnitKernelSingle(
    const unsigned int* coreSetIds,
    const int coreSetSize,
    const int dimension,
    const int minPoints,
    const unsigned long long startIndex,
    const unsigned long long endIndex,
    const unsigned long long minSigBoundary,
    const unsigned long long maxSigBoundary,
    const unsigned long long* labels,
    DenseUnitMap* denseUnitMap,
    BinomialCoeffCreator* binomCoeffs)
{
    unsigned long long x = startIndex + threadIdx.x + blockIdx.x * blockDim.x;

    if (x < endIndex)
    {
        extern __shared__ unsigned int denseUnitIds[];
        int duIdOffset = threadIdx.x * minPoints;

        unsigned long long signature = 0;

        int a = coreSetSize;
        int b = minPoints;

        for (int i = 0; i < minPoints; i++)
        {
            a = binarySearch(x, a - 1, b, binomCoeffs);
            // a = naiveSearch(x, a - 1, b);

            x = x - binomCoeffs->choose(a, b);
            b = b - 1;

            unsigned int id = coreSetIds[a];
            signature += labels[id];
            denseUnitIds[i + duIdOffset] = id;
        }


        if (signature >= minSigBoundary && signature < maxSigBoundary)
        {
            denseUnitMap->insertEntry(signature, denseUnitIds + duIdOffset, dimension);
        }
    }
}


// calculates the next combination sequentially
__device__ int calculateNextCombination(unsigned int* combination, const int minPoints, int pos)
{
    while (pos > 0 && combination[pos] == combination[pos - 1] - 1)
    {
        combination[pos] = minPoints - 1 - pos;
        pos--;
    }

    combination[pos]++;

    if (pos < minPoints - 1)
    {
        pos++;
    }

    return pos;
}

// converts a combination to a dense unit
__device__ unsigned long long combinationToDenseUnit(unsigned int* denseUnitIds, const unsigned int* coreSetIds, const unsigned long long* labels, unsigned int* combination, const int minPoints)
{
    unsigned long long signature = 0;
    for (int i = 0; i < minPoints; i++)
    {
        unsigned int id = coreSetIds[combination[i]];
        signature += labels[id];
        denseUnitIds[i] = id;
    }

    return signature;
}

// kernel for calculating multiple sequantial dense units
__global__ void denseUnitKernelMulti(
    const unsigned int* coreSetIds,
    const int coreSetSize,
    const int dimension,
    const int minPoints,
    const unsigned long long startIndex,
    const unsigned long long endIndex,
    const unsigned long long minSigBoundary,
    const unsigned long long maxSigBoundary,
    const unsigned long long* labels,
    DenseUnitMap* denseUnitMap,
    BinomialCoeffCreator* binomCoeffs,
    int denseUnitsPerThread)
{
    unsigned long long x = startIndex + (threadIdx.x + blockIdx.x * blockDim.x) * denseUnitsPerThread;

    if (x < endIndex)
    {
        // Check if the dense units to calculate fit in the given range
        if (x + denseUnitsPerThread > endIndex)
        {
            denseUnitsPerThread = endIndex - x;
        }

        extern __shared__ unsigned int sharedArray[];

        unsigned int* combination = sharedArray + (threadIdx.x * minPoints * 2);
        unsigned int* denseUnitIds = combination + minPoints;

        unsigned long long signature;

        int a = coreSetSize;
        int b = minPoints;

        for (int i = 0; i < minPoints; i++)
        {
            a = binarySearch(x, a - 1, b, binomCoeffs);
            // a = naiveSearch(x, a - 1, b, binomCoeffs);

            x = x - binomCoeffs->choose(a, b);
            b = b - 1;

            combination[i] = a;
        }

        // Add first dense unit to table (Calculated with Combinadics)
        signature = combinationToDenseUnit(denseUnitIds, coreSetIds, labels, combination, minPoints);
        if (signature >= minSigBoundary && signature < maxSigBoundary)
        {
            denseUnitMap->insertEntry(signature, denseUnitIds, dimension);
        }

        int pos = minPoints - 1;

        for (int i = 0; i < denseUnitsPerThread - 1; i++)
        {
            pos = calculateNextCombination(combination, minPoints, pos);

            // Add the remaining dense unit to table (Calculated sequentially)
            signature = combinationToDenseUnit(denseUnitIds, coreSetIds, labels, combination, minPoints);
            if (signature >= minSigBoundary && signature < maxSigBoundary)
            {
                denseUnitMap->insertEntry(signature, denseUnitIds, dimension);
            }
        }
    }
}

