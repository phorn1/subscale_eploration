#pragma once
#include "../SubscaleTypes.h"
#include "../Tables/DenseUnitTable.cuh"

#include <vector>

// interface for dense unit creators
class IDenseUnitCreator
{
protected:
	DenseUnitTable* denseUnitTable;
	unsigned long long* labels;
	int minPoints;
	int tableSize;

	std::vector<std::vector<CoreSet>> coreSets;

public:
	virtual void calculate(unsigned long long minSigBoundary, unsigned long long maxSigBoundary, int dimension) = 0;
	virtual void createDenseUnits(unsigned long long minSigBoundary, unsigned long long maxSigBoundary) = 0;

	virtual void init(std::vector<std::vector<CoreSet>> coreSets, unsigned long long* labels, int numPoints, int minPoints) = 0;

	virtual void clear() = 0;
	virtual void free() = 0;

	virtual ~IDenseUnitCreator() {};
};