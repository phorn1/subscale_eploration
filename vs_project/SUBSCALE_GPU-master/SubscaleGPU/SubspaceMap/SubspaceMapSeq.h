#pragma once

#include "../Tables/SubspaceTable.cuh"
#include "../DoubleHashing/DoubleHash.h"
#include <cstdint>

class SubspaceMapSeq
{
private:
	DoubleHash<uint32_t>* doubleHash;
	SubspaceTable* table;

public:
	SubspaceMapSeq(SubspaceTable* table, uint32_t tableSize);

	void insertEntry(uint32_t* ids, uint32_t* dimensions);

	void clear();
	void free();
};

