#include "SubspaceTable.cuh"

// adds a dimension to the subspace of an entry
void SubspaceTable::addDimension(unsigned int dimension, unsigned int index)
{
	addValueToBitArray(dimension, getDimensions(index));
}

// adds multiple dimensions to the subspace of an entry
void SubspaceTable::addDimensions(unsigned int* dimensions, unsigned int numberOfDimensions, unsigned int index)
{
	unsigned int* tableDimensions = getDimensions(index);
	for (int i = 0; i < numberOfDimensions; i++)
	{
		addValueToBitArray(dimensions[i], tableDimensions);
	}
}

// converts subspace of an entry from a bitmap to a vector
__host__ vector<unsigned int> SubspaceTable::getDimensionsVec(unsigned int index)
{
	return bitArrayToVector(getDimensions(index), dimensionsSize);
}

// adds a dimension to the subspace of an entry on the device
__device__ void SubspaceTable::dev_addDimension(unsigned int dimension, unsigned int index)
{
	dev_addValueToBitArray(dimension, getDimensions(index));
}

// adds multiple dimensions to the subspace of an entry on the device
__device__ void SubspaceTable::dev_addDimensions(unsigned int* dimensions, unsigned int numberOfDimensions, unsigned int index)
{
	unsigned int* tableDimensions = getDimensions(index);
	for (int i = 0; i < numberOfDimensions; i++)
	{
		dev_addValueToBitArray(dimensions[i], tableDimensions);
	}
}

// adds an id to the ids of an entry
void SubspaceTable::addId(unsigned int id, unsigned int index)
{
	addValueToBitArray(id, getIds(index));
}

// adds multiple ids to the ids of an entry
void SubspaceTable::addIds(unsigned int* ids, unsigned int numberOfIds, unsigned int index)
{
	unsigned int* tableIds = getIds(index);
	for (int i = 0; i < numberOfIds; i++)
	{
		addValueToBitArray(ids[i], tableIds);
	}
}

// converts ids of an entry from a bitmap to a vector
__host__ vector<unsigned int> SubspaceTable::getIdsVec(unsigned int index)
{
	return bitArrayToVector(getIds(index), idsSize);
}

// adds an id to the ids of an entry on the device
__device__ void SubspaceTable::dev_addId(unsigned int id, unsigned int index)
{
	dev_addValueToBitArray(id, getIds(index));
}

// adds multiple ids to the ids of an entry on the device
__device__ void SubspaceTable::dev_addIds(unsigned int* ids, unsigned int numberOfIds, unsigned int index)
{
	unsigned int* tableIds = getIds(index);
	for (int i = 0; i < numberOfIds; i++)
	{
		dev_addValueToBitArray(ids[i], tableIds);
	}
}

// merges ids with the ids of an entry
void SubspaceTable::mergeIds(unsigned int* ids, unsigned int index)
{
	mergeBitArrays(getIds(index), ids, idsSize);
}

// merges ids with the ids of an entry on the device
__device__ void SubspaceTable::dev_mergeIds(unsigned int* ids, unsigned int index)
{
	dev_mergeBitArrays(getIds(index), ids, idsSize);
}
