#pragma once
#include "cuda_runtime.h"
#include "IDenseUnitCreator.h"
#include "../SubscaleTypes.h"
#include "../MemoryManager/MemoryManager.cuh"
#include "../DenseUnitMap/DenseUnitMap.cuh"
#include "../DenseUnitMap/DeviceDenseUnitMap.cuh"
#include "../BinomialCoeffCreator/BinomialCoeffCreator.cuh"
#include "../BinomialCoeffCreator/LocalBinomialCoeffCreator.cuh"
#include "../BinomialCoeffCreator/DeviceBinomialCoeffCreator.cuh"


#include <vector>

// parallel implementation of dense unit creator
class DenseUnitCreator : public IDenseUnitCreator
{
private:
	int numberOfThreads = 64;
	const int numberOfStreams = 1000;

	int denseUnitsPerThread = 5;
	int sharedMemorySize;

	// hash table for dense units
	MemoryManager<DenseUnitMap>* denseUnitMapWrapper;

	// lookup table for binomial coefficients
	LocalBinomialCoeffCreator* local_BinomialCoeffCreator;
	MemoryManager<BinomialCoeffCreator>* binomialCoeffCreatorWrapper;

	cudaStream_t* streams;

	unsigned long long* dev_labels;
	int numDimensions;
	int numPoints;

	unsigned int*** dev_CoreSetIds;

public:
	DenseUnitCreator(DenseUnitTable* denseUnitTable, int tableSize, int numberOfThreads, int denseUnitsPerThread);
	~DenseUnitCreator();

	void calculate(unsigned long long minSigBoundary, unsigned long long maxSigBoundary, int dimension);
	void createDenseUnits(unsigned long long minSigBoundary, unsigned long long maxSigBoundary);

	void init(std::vector<std::vector<CoreSet>> coreSets, unsigned long long* labels,  int numPoints, int minPoints);
	void clear();
	void alloc();
	void copyToDevice();
	void free();
};

