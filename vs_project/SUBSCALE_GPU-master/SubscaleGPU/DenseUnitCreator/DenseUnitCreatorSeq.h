#pragma once
#include "../DenseUnitMap/DenseUnitMapSeq.h"
#include "IDenseUnitCreator.h"
#include "../SubscaleTypes.h"
#include "../Tables/LocalDenseUnitTable.cuh"
#include <vector>

// sequential implementation of dense unit creator
class DenseUnitCreatorSeq : public IDenseUnitCreator
{
private: 

	// hash table for dense units
	DenseUnitMapSeq* denseUnitMap;

	unsigned int* denseUnitIds;
	int* indexStack;
	unsigned long long* signatureStack;

	void denseUnitCalc(
		const unsigned int* coreSetIds,
		const int coreSetSize,
		int pivot,
		const int dimension,
		const unsigned long long minSigBoundary,
		const unsigned long long maxSigBoundary
	);

public:
	DenseUnitCreatorSeq(DenseUnitTable* denseUnitTable, int tableSize);
	~DenseUnitCreatorSeq();

	void calculate(unsigned long long minSigBoundary, unsigned long long maxSigBoundary, int dimension);
	void createDenseUnits(unsigned long long minSigBoundary, unsigned long long maxSigBoundary);

	void init(std::vector<std::vector<CoreSet>> coreSets, unsigned long long* labels, int numPoints, int minPoints);
	void clear();

	void free();
};

