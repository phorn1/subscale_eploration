#pragma once

#include "BinomialCoeffCreator.cuh"
#include "../MemoryManager/MemoryManager.cuh"

// memory manager for the class BinomialCoeffCreator
class BinomialCoeffCreatorMemoryManager : public BinomialCoeffCreator, public virtual MemoryManager<BinomialCoeffCreator>
{
protected:

	BinomialCoeffCreator* allocMembers();
	void freeMembers();

public:
	BinomialCoeffCreatorMemoryManager(unsigned int nMax, unsigned int kMax) : BinomialCoeffCreator(nMax, kMax) {};

	void resetMembers();
};
