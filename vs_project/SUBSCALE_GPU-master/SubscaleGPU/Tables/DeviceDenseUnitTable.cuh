#pragma once

#include "DenseUnitTableMemoryManager.cuh"
#include "../MemoryManager/DeviceMemory.cuh"


class DeviceDenseUnitTable : public DenseUnitTableMemoryManager, public DeviceMemory<DenseUnitTable>
{
public:
	
	DeviceDenseUnitTable(int idsSize, int dimensionsSize, int tableSize) : DenseUnitTableMemoryManager(idsSize, dimensionsSize, tableSize)
	{
		alloc();
		resetMembers();
	}


	~DeviceDenseUnitTable()
	{
		free();
	}
};
