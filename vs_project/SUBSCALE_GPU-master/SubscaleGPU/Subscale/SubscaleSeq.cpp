#include "SubscaleSeq.h"

// calculates cluster candidates from given core sets
LocalSubspaceTable* SubscaleSeq::calculateCandidates(
	vector<vector<CoreSet>> coreSets,
	CsvDataHandler* csvHandler,
	unsigned long long* labels,
	int numberOfDimensions,
	int numberOfPoints,
	unsigned long long minSignature,
	unsigned long long maxSignature)
{
	unsigned int numberOfEntries = 0;

	// start timer
	TimeMeasurement timer;
	timer.start();

	//
	// create local host tables

	// dense unit table
	denseUnitTableWrapper = new LocalDenseUnitTable(config->minPoints, numberOfDimensions, duTableSize);

	// subspace table 1
	localSubspaceTable = new LocalSubspaceTable(numberOfPoints, numberOfDimensions, condensedSsTableSize);

	// subspace table 2
	subspaceTableWrapper = new LocalSubspaceTable(numberOfPoints, numberOfDimensions, ssTableSize);

	// subspace table 3 (same as 1 because one fewer table is needed when all tables are on host memory)
	condensedSsTableWrapper = localSubspaceTable;

	//
	// calculation classes
	DenseUnitCreatorSeq* denseUnitCreator = new DenseUnitCreatorSeq(denseUnitTableWrapper->getPtr(), duTableSize);
	SubspaceJoinerSeq* subspaceJoiner = new SubspaceJoinerSeq(subspaceTableWrapper->getPtr(), ssTableSize);
	TableManagerSeq* tableManager = new TableManagerSeq();

	// initialize
	denseUnitCreator->init(coreSets, labels, numberOfPoints, config->minPoints);
	subspaceJoiner->init(0);
	timer.createTimestamp("Initialisation");

	//
	// calculate slices
	numberOfEntries = calculateAllSlices(denseUnitCreator, subspaceJoiner, csvHandler, tableManager, minSignature, maxSignature);

	// free memory
	delete denseUnitCreator;
	delete denseUnitTableWrapper;
	delete subspaceJoiner;
	delete subspaceTableWrapper;
	delete[] labels;
	coreSets.erase(coreSets.begin(), coreSets.end());
	coreSets.shrink_to_fit();
	timer.createTimestamp("Calculation of Slices");

	// local host table for resulting candidates
	LocalSubspaceTable* resultTable;

	if (config->splittingFactor > 1)
	{
		// if splitting factor is larger than 1, slices have to be read from the filesystem
		ssTableSize = roundToNextPrime(config->finalTableSize);

		// subspace table 4
		subspaceTableWrapper = new LocalSubspaceTable(numberOfPoints, numberOfDimensions, ssTableSize);
		subspaceJoiner = new SubspaceJoinerSeq(subspaceTableWrapper->getPtr(), ssTableSize);
		subspaceJoiner->init(0);

		//
		// combine slices
		numberOfEntries = combineAllSlices(subspaceJoiner, csvHandler, tableManager);

		// subspace table 5
		delete condensedSsTableWrapper;
		condensedSsTableWrapper = new LocalSubspaceTable(numberOfPoints, numberOfDimensions, numberOfEntries);

		tableManager->condenseTable(condensedSsTableWrapper->getPtr(), subspaceTableWrapper->getPtr(), ssTableSize);
		delete subspaceJoiner;
		delete subspaceTableWrapper;

	}

	// subspace table 6
	resultTable = new LocalSubspaceTable(numberOfPoints, numberOfDimensions, numberOfEntries);
	tableManager->localToLocal(resultTable, condensedSsTableWrapper->getPtr(), numberOfEntries);
	
	// free memory
	delete condensedSsTableWrapper;
	delete tableManager;
	timer.createTimestamp("Combination of slices");

	// write time differences to an output file
	std::string filePath = config->resultPath + "time_Subscale.txt";
	timer.writeTimestampDeltas(filePath.c_str());

	return resultTable;
}