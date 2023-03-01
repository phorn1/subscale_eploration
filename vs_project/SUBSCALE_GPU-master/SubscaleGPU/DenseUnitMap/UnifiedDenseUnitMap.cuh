#pragma once

#include "DenseUnitMapMemoryManager.cuh"
#include "../MemoryManager/UnifiedMemory.cuh"


class UnifiedDenseUnitMap : public DenseUnitMapMemoryManager, public UnifiedMemory<DenseUnitMap>
{
public:

	UnifiedDenseUnitMap(DenseUnitTable* table, int tableSize, unsigned int ticketSize)
		: DenseUnitMapMemoryManager(table, tableSize, ticketSize)
	{
		alloc();
		resetMembers();
	}


	~UnifiedDenseUnitMap()
	{
		free();
	}
};