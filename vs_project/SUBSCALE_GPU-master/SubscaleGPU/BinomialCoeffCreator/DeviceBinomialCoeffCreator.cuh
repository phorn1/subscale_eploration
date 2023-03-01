#pragma once

#include "BinomialCoeffCreatorMemoryManager.cuh"
#include "../MemoryManager/DeviceMemory.cuh"


class DeviceBinomialCoeffCreator : public BinomialCoeffCreatorMemoryManager, public DeviceMemory<BinomialCoeffCreator>
{
public:

	DeviceBinomialCoeffCreator(unsigned int nMax, unsigned int kMax) : BinomialCoeffCreatorMemoryManager(nMax, kMax)
	{
		alloc();
		resetMembers();
	}


	~DeviceBinomialCoeffCreator()
	{
		free();
	}
};