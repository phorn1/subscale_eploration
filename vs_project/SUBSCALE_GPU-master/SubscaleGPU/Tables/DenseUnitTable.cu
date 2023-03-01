#include "DenseUnitTable.cuh"

// adds a dimension to the subspace of a dense unit
__host__ void DenseUnitTable::addDimension(unsigned int dimension, unsigned int index)
{
	addValueToBitArray(dimension, getDimensions(index));
}

// adds multiple dimensions to the subspace of a dense unit
__host__ void DenseUnitTable::addDimensions(unsigned int* dimensions, unsigned int numberOfDimensions, unsigned int index)
{
	unsigned int* tableDimensions = getDimensions(index);
	for (int i = 0; i < numberOfDimensions; i++)
	{
		addValueToBitArray(dimensions[i], tableDimensions);
	}
}

// converts subspace of a dense unit from a bitmap to a vector
__host__ vector<unsigned int> DenseUnitTable::getDimensionsVec(unsigned int index)
{
	return bitArrayToVector(getDimensions(index), dimensionsSize);
}

// adds a dimension to the subspace of a dense unit on the device
__device__ void DenseUnitTable::dev_addDimension(unsigned int dimension, unsigned int index)
{
	dev_addValueToBitArray(dimension, getDimensions(index));
}

// adds multiple dimensions to the subspace of a dense unit on the device
__device__ void DenseUnitTable::dev_addDimensions(unsigned int* dimensions, unsigned int numberOfDimensions, unsigned int index)
{
	unsigned int* tableDimensions = getDimensions(index);
	for (int i = 0; i < numberOfDimensions; i++)
	{
		dev_addValueToBitArray(dimensions[i], tableDimensions);
	}
}