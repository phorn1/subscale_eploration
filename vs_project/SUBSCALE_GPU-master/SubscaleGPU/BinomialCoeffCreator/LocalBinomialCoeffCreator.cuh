#pragma once

#include "BinomialCoeffCreatorMemoryManager.cuh"
#include "../MemoryManager/LocalMemory.cuh"


class LocalBinomialCoeffCreator : public BinomialCoeffCreatorMemoryManager, public LocalMemory<BinomialCoeffCreator>
{
public:

	LocalBinomialCoeffCreator(unsigned int nMax, unsigned int kMax) : BinomialCoeffCreatorMemoryManager(nMax, kMax)
	{
		alloc();
		resetMembers();
	}


	~LocalBinomialCoeffCreator()
	{
		free();
	}
};