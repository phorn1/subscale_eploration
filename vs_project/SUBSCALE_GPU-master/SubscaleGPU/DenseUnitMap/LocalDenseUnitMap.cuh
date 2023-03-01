#pragma once

#include "DenseUnitMapMemoryManager.cuh"
#include "../MemoryManager/LocalMemory.cuh"


class LocalDenseUnitMap : public DenseUnitMapMemoryManager, public LocalMemory<DenseUnitMap>
{
public:

	LocalDenseUnitMap(DenseUnitTable* table, int tableSize, unsigned int ticketSize)
		: DenseUnitMapMemoryManager(table, tableSize, ticketSize)
	{
		alloc();
		resetMembers();
	}


	~LocalDenseUnitMap()
	{
		free();
	}
};