#pragma once

#include "SubspaceMap.cuh"
#include "../MemoryManager/MemoryManager.cuh"

class SubspaceMapMemoryManager : public SubspaceMap, public virtual MemoryManager<SubspaceMap>
{
protected:

	SubspaceMap* allocMembers();
	void freeMembers();

public:
	SubspaceMapMemoryManager(SubspaceTable* table, uint32_t tableSize, uint32_t ticketSize) 
		: SubspaceMap(table, tableSize, ticketSize) {}

	void resetMembers();
};


