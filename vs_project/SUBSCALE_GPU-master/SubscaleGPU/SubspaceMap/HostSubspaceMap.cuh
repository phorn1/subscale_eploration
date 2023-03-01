#pragma once

#include "SubspaceMapMemoryManager.cuh"
#include "../MemoryManager/HostMemory.cuh"


class HostSubspaceMap : public SubspaceMapMemoryManager, public HostMemory<SubspaceMap>
{
public:

	HostSubspaceMap(SubspaceTable* table, uint32_t tableSize, uint32_t ticketSize)
		: SubspaceMapMemoryManager(table, tableSize, ticketSize)
	{
		alloc();
		resetMembers();
	}


	~HostSubspaceMap()
	{
		free();
	}
};