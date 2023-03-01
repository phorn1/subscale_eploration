#pragma once

#include "SubspaceTableMemoryManager.cuh"
#include "../MemoryManager/DeviceMemory.cuh"

class DeviceSubspaceTable : public SubspaceTableMemoryManager, public DeviceMemory<SubspaceTable>
{
public:

	DeviceSubspaceTable(int idsSize, int dimensionsSize, int tableSize) : SubspaceTableMemoryManager(idsSize, dimensionsSize, tableSize)
	{
		alloc();
		resetMembers();
	}

	~DeviceSubspaceTable()
	{
		free();
	}
};