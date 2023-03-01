#pragma once
#include "cuda_runtime.h"

#include "../StadiumHashing/StadiumHash.cuh"
#include "../Tables/DenseUnitTable.cuh"

#include <stdint.h>

// parallel implementation of a hashtable for dense units
class DenseUnitMap : public StadiumHash<uint64_t, uint32_t>
{
protected:
	uint64_t* keys;
	DenseUnitTable* table;

public:
	DenseUnitMap(DenseUnitTable* table, uint32_t tableSize, uint32_t ticketSize);

	__device__ void insertEntry(uint64_t signature, uint32_t* ids, uint32_t dimension);
};