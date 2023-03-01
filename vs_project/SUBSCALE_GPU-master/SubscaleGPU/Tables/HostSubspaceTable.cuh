#pragma once

#include "SubspaceTableMemoryManager.cuh"
#include "HostMemory.cuh"

class HostSubspaceTable : public SubspaceTableMemoryManager, public HostMemory<SubspaceTable>
{
public:

	HostSubspaceTable(int idsSize, int dimensionsSize, int tableSize) : SubspaceTableMemoryManager(idsSize, dimensionsSize, tableSize)
	{
		alloc();
		resetMembers();
	}

	~HostSubspaceTable()
	{
		free();
	}
};