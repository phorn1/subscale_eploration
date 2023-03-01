#pragma once

#include "BinomialCoeffCreatorMemoryManager.cuh"
#include "Table/UnifiedMemory.cuh"


class UnifiedBinomialCoeffCreator : public BinomialCoeffCreatorMemoryManager, public UnifiedMemory<BinomialCoeffCreator>
{
public:

	UnifiedBinomialCoeffCreator(unsigned int nMax, unsigned int kMax) : BinomialCoeffCreatorMemoryManager(nMax, kMax)
	{
		alloc();
		resetMembers();
	}


	~UnifiedBinomialCoeffCreator()
	{
		free();
	}
};
