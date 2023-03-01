#pragma once

#include "cuda_runtime.h"
#include <stdio.h>
#include <assert.h>

//
// Basic table for the subscale algorithm. It contains one column for the ids of the points and one column for dimensions.
// Ids and dimensions can contain multiple values in each row. The number of values is determined by idsSize and
// dimensionsSize respectively. 
class SubscaleTable {
protected:
	unsigned int* ids;
	unsigned int* dimensions;

	int tableSize;
	int idsSize;
	int dimensionsSize;

public:

	__host__ SubscaleTable(int idsSize, int dimensionsSize, int tableSize);

	__host__ SubscaleTable(SubscaleTable* table);

	__host__ __device__ void insertIds(unsigned int* ids, unsigned int index);

	__host__ __device__ void setIdsZero(unsigned int index);

	__host__ __device__ unsigned int* getIds();

	__host__ __device__ unsigned int* getIds(unsigned int index);

	__host__ __device__ int getIdsSize();

	__host__ __device__ void insertDimensions(unsigned int* dimensions, unsigned int index);

	__host__ __device__ void setDimensionsZero(unsigned int index);

	__host__ __device__ unsigned int* getDimensions();

	__host__ __device__ unsigned int* getDimensions(unsigned int index);

	__host__ __device__ int getDimensionsSize();

	__host__ __device__ int getTableSize();

	__host__ __device__ void printTable();
};



