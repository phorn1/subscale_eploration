#pragma once

#include "SubspaceTableMemoryManager.cuh"
#include "../MemoryManager/LocalMemory.cuh"

class LocalSubspaceTable : public SubspaceTableMemoryManager, public LocalMemory<SubspaceTable>
{
public:

	LocalSubspaceTable(int idsSize, int dimensionsSize, int tableSize) : SubspaceTableMemoryManager(idsSize, dimensionsSize, tableSize)
	{
		alloc();
		resetMembers();
	}

	~LocalSubspaceTable()
	{
		free();
	}
};