#pragma once

#include "SubspaceTableMemoryManager.cuh"
#include "UnifiedMemory.cuh"

class UnifiedSubspaceTable : public SubspaceTableMemoryManager, public UnifiedMemory<SubspaceTable>
{
public:

	UnifiedSubspaceTable(int idsSize, int dimensionsSize, int tableSize) : SubspaceTableMemoryManager(idsSize, dimensionsSize, tableSize)
	{
		alloc();
		resetMembers();
	}

	~UnifiedSubspaceTable()
	{
		free();
	}
};