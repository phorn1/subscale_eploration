#pragma once
#include <vector>
#include "cuda_runtime.h"

using namespace std;


// Class to map unsigned integers to bits. The bits are stored in bit arrays consisting of multiple 32-bit unsigned integers.
class BitMapper {
protected:
	__host__ __device__ unsigned int getIndexInBitArray(unsigned int value);
	__host__ __device__ unsigned int getPosInBitField(unsigned int value);
	vector<unsigned int> bitArrayToVector(unsigned int* bitArray, unsigned int arraySize);

	__host__ void addValueToBitArray(unsigned int value, unsigned int* bitArray);
	__host__ void mergeBitArrays(unsigned int* targetArray, unsigned int* sourceArray, unsigned int arraySize);
	__device__ void dev_addValueToBitArray(unsigned int value, unsigned int* bitArray);
	__device__ void dev_mergeBitArrays(unsigned int* targetArray, unsigned int* sourceArray, unsigned int arraySize);
};