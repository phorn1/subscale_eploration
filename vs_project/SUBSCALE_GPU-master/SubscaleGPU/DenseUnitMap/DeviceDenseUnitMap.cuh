#pragma once

#include "DenseUnitMapMemoryManager.cuh"
#include "../MemoryManager/DeviceMemory.cuh"



class DeviceDenseUnitMap : public DenseUnitMapMemoryManager, public DeviceMemory<DenseUnitMap>
{
public:

	DeviceDenseUnitMap(DenseUnitTable* table, int tableSize, unsigned int ticketSize) 
		: DenseUnitMapMemoryManager(table, tableSize, ticketSize)
	{
		alloc();
		resetMembers();
	}


	~DeviceDenseUnitMap()
	{
		free();
	}
};