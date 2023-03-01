#pragma once

#include "DenseUnitMapMemoryManager.cuh"
#include "../MemoryManager/HostMemory.cuh"


class HostDenseUnitMap : public DenseUnitMapMemoryManager, public HostMemory<DenseUnitMap>
{
public:

	HostDenseUnitMap(DenseUnitTable* table, int tableSize, unsigned int ticketSize)
		: DenseUnitMapMemoryManager(table, tableSize, ticketSize)
	{
		alloc();
		resetMembers();
	}


	~HostDenseUnitMap()
	{
		free();
	}
};