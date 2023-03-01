#pragma once

#include "BinomialCoeffCreatorMemoryManager.cuh"
#include "Table/HostMemory.cuh"


class HostBinomialCoeffCreator : public BinomialCoeffCreatorMemoryManager, public HostMemory<BinomialCoeffCreator>
{
public:

	HostBinomialCoeffCreator(unsigned int nMax, unsigned int kMax) : BinomialCoeffCreatorMemoryManager(nMax, kMax)
	{
		alloc();
		resetMembers();
	}


	~HostBinomialCoeffCreator()
	{
		free();
	}
};