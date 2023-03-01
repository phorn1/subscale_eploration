#pragma once

#include "DenseUnitTableMemoryManager.cuh"
#include "../MemoryManager/LocalMemory.cuh"

class LocalDenseUnitTable : public DenseUnitTableMemoryManager, public LocalMemory<DenseUnitTable>
{
public:

	LocalDenseUnitTable(int idsSize, int dimensionsSize, int tableSize) : DenseUnitTableMemoryManager(idsSize, dimensionsSize, tableSize)
	{
		alloc();
		resetMembers();
	}

	~LocalDenseUnitTable()
	{
		free();
	}
};