#include "SubscaleTable.cuh"

// Constructor
__host__ SubscaleTable::SubscaleTable(int idsSize, int dimensionsSize, int tableSize)
{
	this->idsSize = idsSize;
	this->dimensionsSize = dimensionsSize;
	this->tableSize = tableSize;
}

// Copy constructor
__host__ SubscaleTable::SubscaleTable(SubscaleTable* table)
{
	this->idsSize = table->getIdsSize();
	this->dimensionsSize = table->getDimensionsSize();
	this->tableSize = table->getTableSize();
}

// Inserts the ids into the entry with the given index
__host__ __device__ void SubscaleTable::insertIds(unsigned int* ids, unsigned int index)
{
	// Check if index fits in the table
	assert(("Index for table id insertion out of bounds!", index < tableSize));

	// Insert ids into the table
	unsigned long long arrIndex = index * idsSize;
	for (int i = 0; i < idsSize; i++)
	{
		this->ids[arrIndex + i] = ids[i];
	}
}

// Sets all ids of the entry with the given index to 0
__host__ __device__ void SubscaleTable::setIdsZero(unsigned int index)
{
	// Check if index fits in the table
	assert(("Table index out of bounds!", index < tableSize));

	// Insert ids into the table
	unsigned long long arrIndex = index * idsSize;
	for (int i = 0; i < idsSize; i++)
	{
		this->ids[arrIndex + i] = 0;
	}
}

// Get ids size
__host__ __device__ unsigned int* SubscaleTable::getIds() {
	return ids;
}

// Get ids of the entry with the given index
__host__ __device__ unsigned int* SubscaleTable::getIds(unsigned int index) {
	return ids + ((unsigned long long)index * idsSize);
}

// Get number of ids that are stored in one entry
__host__ __device__ int SubscaleTable::getIdsSize() {
	return idsSize;
}

// Inserts the dimensions into the entry with the given index
__host__ __device__ void SubscaleTable::insertDimensions(unsigned int* dimensions, unsigned int index)
{
	// Check if index fits in the table
	assert(("Index for table dimensoins insertion out of bounds!\n", index < tableSize));

	// Insert dimensions into the table
	unsigned long long arrIndex = index * dimensionsSize;
	for (int i = 0; i < dimensionsSize; i++)
	{
		this->dimensions[arrIndex + i] = dimensions[i];
	}
}

// Sets all dimensions of the entry with the given index to 0
__host__ __device__ void SubscaleTable::setDimensionsZero(unsigned int index)
{
	// Check if index fits in the table
	assert(("Table index out of bounds!", index < tableSize));

	// Insert ids into the table
	unsigned long long arrIndex = index * dimensionsSize;
	for (int i = 0; i < dimensionsSize; i++)
	{
		this->dimensions[arrIndex + i] = 0;
	}
}

// Get the dimensions column
__host__ __device__ unsigned int* SubscaleTable::getDimensions()
{
	return dimensions;
}

// Get dimensions of the entry with the given index
__host__ __device__ unsigned int* SubscaleTable::getDimensions(unsigned int index) {
	return dimensions + ((unsigned long long)index * dimensionsSize);
}

// Get dimensions size
__host__ __device__ int SubscaleTable::getDimensionsSize() {
	return dimensionsSize;
}

// Get table size
__host__ __device__ int SubscaleTable::getTableSize() {
	return tableSize;
}

// Print all entries from the table
__host__ __device__ void SubscaleTable::printTable()
{
	for (int i = 0; i < tableSize; i++)
	{
		printf("%d.\nIDs: ", i);
		for (int j = 0; j < idsSize; j++)
		{
			printf("%u ", getIds(i)[j]);
		}

		printf("\nDimensions: ");
		for (int j = 0; j < dimensionsSize; j++)
		{
			printf("%u ", getDimensions(i)[j]);
		}

		printf("\n");
	}
}




