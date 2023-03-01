#include "Subscale.h"

#include "../TimeMeasurement/TimeMeasurement.h"

#define TICKET_SIZE 4

// calculates cluster candidates from given core sets
LocalSubspaceTable* Subscale::calculateCandidates(
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
	// create device tables

	// dense unit table
	denseUnitTableWrapper = new DeviceDenseUnitTable(config->minPoints, numberOfDimensions, duTableSize);

	// subspace table 1
	condensedSsTableWrapper = new DeviceSubspaceTable(numberOfPoints, numberOfDimensions, condensedSsTableSize);

	// subspace table 2
	subspaceTableWrapper = new DeviceSubspaceTable(numberOfPoints, numberOfDimensions, ssTableSize);

	// subspace table 3
	localSubspaceTable = new LocalSubspaceTable(numberOfPoints, numberOfDimensions, condensedSsTableSize);
	
	//
	// calculation classes
	DenseUnitCreator* denseUnitCreator = new DenseUnitCreator(denseUnitTableWrapper->getPtr(), duTableSize, config->threadsPerBlock, config->denseUnitsPerThread);
	SubspaceJoiner* subspaceJoiner = new SubspaceJoiner(subspaceTableWrapper->getPtr(), ssTableSize, config->threadsPerBlock);
	TableManager* tableManager = new TableManager(config->threadsPerBlock);

	// initialize
	denseUnitCreator->init(coreSets, labels, numberOfPoints, config->minPoints);
	subspaceJoiner->init(TICKET_SIZE);
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
		subspaceTableWrapper = new DeviceSubspaceTable(numberOfPoints, numberOfDimensions, ssTableSize);
		subspaceJoiner = new SubspaceJoiner(subspaceTableWrapper->getPtr(), ssTableSize, config->threadsPerBlock);
		subspaceJoiner->init(TICKET_SIZE);

		//
		// combine slices
		numberOfEntries = combineAllSlices(subspaceJoiner, csvHandler, tableManager);
	
		// subspace table 5
		delete condensedSsTableWrapper;
		condensedSsTableWrapper = new DeviceSubspaceTable(numberOfPoints, numberOfDimensions, numberOfEntries);
	

		tableManager->condenseTable(condensedSsTableWrapper->getPtr(), subspaceTableWrapper->getPtr(), ssTableSize);
		delete subspaceJoiner;
		delete subspaceTableWrapper;

		// subspace table 6
		resultTable = new LocalSubspaceTable(numberOfPoints, numberOfDimensions, numberOfEntries);
		tableManager->deviceToLocal(resultTable, condensedSsTableWrapper->getPtr(), numberOfEntries);
	}
	else
	{
		// if splitting factor is 1, there is only 1 slice that is already in the host memory
		resultTable = new LocalSubspaceTable(numberOfPoints, numberOfDimensions, numberOfEntries);
		tableManager->localToLocal(resultTable, localSubspaceTable, numberOfEntries);
	}

	// free memory
	delete localSubspaceTable;
	delete condensedSsTableWrapper;
	delete tableManager;
	timer.createTimestamp("Combination of slices");

	// write time differences to an output file
	std::string filePath = config->resultPath + "time_Subscale.txt";
	timer.writeTimestampDeltas(filePath.c_str());

	return resultTable;
}






