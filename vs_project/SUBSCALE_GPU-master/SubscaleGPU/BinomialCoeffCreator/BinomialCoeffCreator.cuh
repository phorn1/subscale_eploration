#pragma once
#include "cuda_runtime.h"

// class to create binomial coefficients
class BinomialCoeffCreator
{
protected:
	unsigned int nMax;
	unsigned int kMax;

	unsigned long long* bcArray;
	unsigned long long numberOfBCs;

	__host__ __device__ void addBinomCoeff(unsigned long long* arr, unsigned int n, unsigned int k, unsigned long long value);
public:
	BinomialCoeffCreator(unsigned int nMax, unsigned int kMax);

	__host__ __device__ unsigned long long choose(unsigned int n, unsigned int k);
	__host__ __device__ unsigned long long* getArray();
	__host__ __device__ void calculateBCs();
	__host__ __device__ void calculateBCs(unsigned long long* arr);
	__host__ __device__ void print();
};