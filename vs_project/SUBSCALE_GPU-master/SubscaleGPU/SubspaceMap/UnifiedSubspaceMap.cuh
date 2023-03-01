#pragma once

#include "SubspaceMapMemoryManager.cuh"
#include "../MemoryManager/UnifiedMemory.cuh"


class UnifiedSubspaceMap : public SubspaceMapMemoryManager, public UnifiedMemory<SubspaceMap>
{
public:

	UnifiedSubspaceMap(SubspaceTable* table, uint32_t tableSize, uint32_t ticketSize)
		: SubspaceMapMemoryManager(table, tableSize, ticketSize)
	{
		alloc();
		resetMembers();
	}


	~UnifiedSubspaceMap()
	{
		free();
	}
};