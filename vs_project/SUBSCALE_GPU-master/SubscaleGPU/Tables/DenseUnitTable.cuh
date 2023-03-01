#pragma once
#include "SubscaleTable.cuh"
#include "BitMapper.cuh" 
#include "../HelperFunctions/roundingFunctions.h"

#include <limits>

// Class to create a table for dense units
class DenseUnitTable : public SubscaleTable, public BitMapper {
public:
	DenseUnitTable(int idsSize, int dimensionsSize, int tableSize) : SubscaleTable(idsSize, dimensionsSize, tableSize)
	{
		this->dimensionsSize = roundToNextQuotient(dimensionsSize, sizeof(unsigned int) * CHAR_BIT);
	}

	__host__ void addDimension(unsigned int dimension, unsigned int index);

	__host__ void addDimensions(unsigned int* dimensions, unsigned int numberOfDimensions, unsigned int index);

	__host__ vector<unsigned int> getDimensionsVec(unsigned int index);

	__device__ void dev_addDimension(unsigned int dimension, unsigned int index);

	__device__ void dev_addDimensions(unsigned int* dimensions, unsigned int numberOfDimensions, unsigned int index);
};