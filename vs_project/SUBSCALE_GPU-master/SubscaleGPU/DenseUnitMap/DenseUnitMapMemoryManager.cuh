#pragma once

#include "DenseUnitMap.cuh"
#include "../MemoryManager/MemoryManager.cuh"

// memory manager for the class DenseUnitMap
class DenseUnitMapMemoryManager : public DenseUnitMap, public virtual MemoryManager<DenseUnitMap>
{
protected:

	DenseUnitMap* allocMembers();
	void freeMembers();

public:
	DenseUnitMapMemoryManager(DenseUnitTable* table, uint32_t tableSize, uint32_t ticketSize) 
		: DenseUnitMap(table, tableSize, ticketSize) {};

	void resetMembers();
};


