#pragma once
#include "SubscaleTable.cuh"
#include "BitMapper.cuh"
#include "../HelperFunctions/roundingFunctions.h"

#include <limits>

class SubspaceTable : public SubscaleTable, public BitMapper {
public:
	SubspaceTable(int idsSize, int dimensionsSize, int tableSize) : SubscaleTable(idsSize, dimensionsSize, tableSize) 
	{
		this->idsSize = roundToNextQuotient(idsSize, sizeof(unsigned int) * CHAR_BIT);
		this->dimensionsSize = roundToNextQuotient(dimensionsSize, sizeof(unsigned int) * CHAR_BIT);
	}

	__host__ void addDimension(unsigned int dimension, unsigned int index);
	__host__ void addDimensions(unsigned int* dimensions, unsigned int numberOfDimensions, unsigned int index);
	__host__ vector<unsigned int> getDimensionsVec(unsigned int index);
	__device__ void dev_addDimension(unsigned int dimension, unsigned int index);
	__device__ void dev_addDimensions(unsigned int* dimensions, unsigned int numberOfDimensions, unsigned int index);
	
	__host__ void addId(unsigned int id, unsigned int index);
	__host__ void addIds(unsigned int* ids, unsigned int numberOfIds, unsigned int index);
	__host__ vector<unsigned int> getIdsVec(unsigned int index);
	__device__ void dev_addId(unsigned int id, unsigned int index);
	__device__ void dev_addIds(unsigned int* ids, unsigned int numberOfIds, unsigned int index);

	__host__ void mergeIds(unsigned int* ids, unsigned int index);
	__device__ void dev_mergeIds(unsigned int* ids, unsigned int index);
};