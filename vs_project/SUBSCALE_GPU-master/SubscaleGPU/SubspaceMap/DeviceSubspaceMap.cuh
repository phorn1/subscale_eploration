#pragma once

#include "SubspaceMapMemoryManager.cuh"
#include "../MemoryManager/DeviceMemory.cuh"


class DeviceSubspaceMap : public SubspaceMapMemoryManager, public DeviceMemory<SubspaceMap>
{
public:

	DeviceSubspaceMap(SubspaceTable* table, uint32_t tableSize, uint32_t ticketSize) 
		: SubspaceMapMemoryManager(table, tableSize, ticketSize)
	{
		alloc();
		resetMembers();
	}

	~DeviceSubspaceMap()
	{
		free();
	}
};