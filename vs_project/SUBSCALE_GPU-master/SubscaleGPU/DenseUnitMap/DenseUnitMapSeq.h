#pragma once
#include "../Tables/DenseUnitTable.cuh"
#include "../DoubleHashing/DoubleHash.h"
#include <cstdint>

// sequential implementation of a hashtable for dense units
class DenseUnitMapSeq
{
private:
	int tableSize;
	uint64_t* keys;

	DoubleHash<uint64_t>* doubleHash;
	DenseUnitTable* table;

	void addEntry(uint32_t hashed, uint64_t signature, uint32_t* ids, uint32_t dimension);
	void addDimension(uint32_t hashed, uint32_t dimension);
public:

	DenseUnitMapSeq(DenseUnitTable* denseUnitTable, uint32_t tableSize);
	~DenseUnitMapSeq();
	void insertDenseUnit(uint64_t signature, uint32_t* ids, uint32_t dimension);

	void clear();
	void free();

};

