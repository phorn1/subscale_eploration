#pragma once

#include "DenseUnitTableMemoryManager.cuh"
#include "HostMemory.cuh"


class HostDenseUnitTable : public DenseUnitTableMemoryManager, public HostMemory<DenseUnitTable>
{
public:

	HostDenseUnitTable(int idsSize, int dimensionsSize, int tableSize) : DenseUnitTableMemoryManager(idsSize, dimensionsSize, tableSize)
	{
		alloc();
		resetMembers();
	}


	~HostDenseUnitTable()
	{
		free();
	}
};