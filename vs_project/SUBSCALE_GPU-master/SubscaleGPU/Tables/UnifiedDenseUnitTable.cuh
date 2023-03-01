#pragma once

#include "DenseUnitTableMemoryManager.cuh"
#include "UnifiedMemory.cuh"


class UnifiedDenseUnitTable : public DenseUnitTableMemoryManager, public UnifiedMemory<DenseUnitTable>
{
public:

	UnifiedDenseUnitTable(int idsSize, int dimensionsSize, int tableSize) : DenseUnitTableMemoryManager(idsSize, dimensionsSize, tableSize)
	{
		alloc();
		resetMembers();
	}


	~UnifiedDenseUnitTable()
	{
		free();
	}
};