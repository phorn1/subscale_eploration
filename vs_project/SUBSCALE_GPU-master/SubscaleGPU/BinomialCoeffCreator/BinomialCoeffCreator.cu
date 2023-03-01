#include "BinomialCoeffCreator.cuh"
#include <stdio.h>

// Adds a binomial coefficent to the array
__host__ __device__ void BinomialCoeffCreator::addBinomCoeff(unsigned long long* arr, unsigned int n, unsigned int k, unsigned long long value)
{
	arr[(k - 1) * (nMax + 1) + n] = value;
}

// Constructor
BinomialCoeffCreator::BinomialCoeffCreator(unsigned int nMax, unsigned int kMax)
{
	this->nMax = nMax;
	this->kMax = kMax;
	this->numberOfBCs = (nMax + 1) * kMax;
}

// Returns the binomial coefficient "n choose k" from the array
__host__ __device__ unsigned long long BinomialCoeffCreator::choose(unsigned int n, unsigned int k)
{
	return bcArray[(k - 1) * (nMax + 1) + n];
}

// Get the array containing all binomial coefficients
__host__ __device__ unsigned long long* BinomialCoeffCreator::getArray()
{
	return bcArray;
}

// Calculate all possible binomial coefficients with 0 <= n <= nMax and 1 <= k <= kMax
__host__ __device__ void BinomialCoeffCreator::calculateBCs()
{
	calculateBCs(bcArray);
}

// Calculate all possible binomial coefficients and store them in the given array
__host__ __device__ void BinomialCoeffCreator::calculateBCs(unsigned long long* arr)
{
	int kTemp;
	for (int k = 1; k <= kMax; k++)
	{
		for (int n = 0; n <= nMax; n++)
		{
			if (n < k)
			{
				addBinomCoeff(arr, n, k, 0);
			}
			else
			{
				if (2 * k > n)
				{
					kTemp = n - k;
				}
				else
				{
					kTemp = k;
				}

				unsigned long long bc = 1;

				for (int i = 1; i <= kTemp; i++)
				{
					bc = bc * ((unsigned long long) n - kTemp + i) / i;
				}

				addBinomCoeff(arr, n, k, bc);
			}
		}
	}
}

// Prints all binomial coefficients from the array
__host__ __device__ void BinomialCoeffCreator::print()
{
	for (int k = 1; k <= kMax; k++)
	{
		for (int n = 0; n <= nMax; n++)
		{
			printf("%d choose %d = %llu\n", n, k, choose(n, k));
		}
	}
}
