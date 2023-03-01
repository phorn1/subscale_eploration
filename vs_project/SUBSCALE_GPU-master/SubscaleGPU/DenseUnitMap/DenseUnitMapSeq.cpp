#include "DenseUnitMapSeq.h"
#include <cstring>

// adds a new entry
void DenseUnitMapSeq::addEntry(uint32_t hashed, uint64_t signature, uint32_t* ids, uint32_t dimension)
{
	keys[hashed] = signature;

	table->insertIds(ids, hashed);
	table->addDimension(dimension, hashed);
}

// adds dimension to an existing entry
void DenseUnitMapSeq::addDimension(uint32_t index, uint32_t dimension)
{
	table->addDimension(dimension, index);
}

// constructor
DenseUnitMapSeq::DenseUnitMapSeq(DenseUnitTable* denseUnitTable, uint32_t tableSize)
{
	this->tableSize = tableSize;
	keys = new uint64_t[tableSize];
	doubleHash = new DoubleHash<uint64_t>(tableSize);
	table = denseUnitTable;
}

// destructor
DenseUnitMapSeq::~DenseUnitMapSeq()
{
	free();
}

// inserts a dense unit into the hash map
void DenseUnitMapSeq::insertDenseUnit(uint64_t signature, uint32_t* ids, uint32_t dimension)
{
	std::pair<bool, int> got = doubleHash->find(signature, this->keys);

	if (got.first)
	{
		// if signature is found in table, add dimension to the found entry
		addDimension(got.second, dimension);
	}
	else
	{
		// if signature is not found in table, add new entry to the table
		addEntry(got.second, signature, ids, dimension);
	}
}

// clears hash map
void DenseUnitMapSeq::clear()
{
	memset(keys, 0, tableSize * sizeof(uint64_t));
}

// frees memory
void DenseUnitMapSeq::free()
{
	delete[] keys;
	delete doubleHash;
}

