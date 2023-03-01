#pragma once
#include "../StadiumHashing/ConcStadiumHash.cuh"
#include "../Tables/SubspaceTable.cuh"
#include <cstdint>

// Class to create a table for cluster candidates
class SubspaceMap : public ConcStadiumHash<uint32_t, uint32_t>
{
protected:
	SubspaceTable* table;

public:
	
	SubspaceMap(SubspaceTable* table, uint32_t tableSize, uint32_t ticketSize);

	__device__ void insertEntry(uint32_t* ids, uint32_t* dimensions);
};

