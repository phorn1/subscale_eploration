#include "BinomialCoeffCreatorMemoryManager.cuh"

BinomialCoeffCreator* BinomialCoeffCreatorMemoryManager::allocMembers()
{
	// Allocate array for binomial coefficients
	allocArray((uint64_t**) &bcArray, numberOfBCs);

	return (BinomialCoeffCreator*) this;
}

void BinomialCoeffCreatorMemoryManager::freeMembers()
{
	// Free array for binomial coefficients
	freeArray((uint64_t*) bcArray);
}

void BinomialCoeffCreatorMemoryManager::resetMembers()
{
	// Calculate binomial coefficients locally
	unsigned long long* binomialCoeffs = new unsigned long long[numberOfBCs];
	calculateBCs(binomialCoeffs);

	// Copy binomial coefficients to to a member variable
	copyArrayContent((uint64_t*) bcArray, (uint64_t*) binomialCoeffs, numberOfBCs);

	// Delete local binomial coefficients
	delete[] binomialCoeffs;
}