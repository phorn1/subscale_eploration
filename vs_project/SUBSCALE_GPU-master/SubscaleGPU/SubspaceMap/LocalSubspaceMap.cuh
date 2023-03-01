#pragma once

#include "SubspaceMapMemoryManager.cuh"
#include "../MemoryManager/LocalMemory.cuh"


class LocalSubspaceMap : public SubspaceMapMemoryManager, public LocalMemory<SubspaceMap>
{
public:

	LocalSubspaceMap(SubspaceTable* table, uint32_t tableSize, uint32_t ticketSize)
		: SubspaceMapMemoryManager(table, tableSize, ticketSize)
	{
		alloc();
		resetMembers();
	}


	~LocalSubspaceMap()
	{
		free();
	}
};