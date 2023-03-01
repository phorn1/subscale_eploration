#include "ISubscale.h"

// Count the total number of dense units that are created from the core sets
unsigned long long ISubscale::countPossibleDenseUnits(vector<vector<CoreSet>> coreSets, int numPoints, int minPoints)
{
	LocalBinomialCoeffCreator binomCoeffs(numPoints, minPoints);

	unsigned long long count = 0;

	for (vector<CoreSet> coreSetsInDimension : coreSets)
	{
		for (CoreSet coreSet : coreSetsInDimension)
		{
			unsigned long long startIndex = binomCoeffs.choose(coreSet.pivot + 1, minPoints);
			unsigned long long endIndex = binomCoeffs.choose(coreSet.size, minPoints);

			count += (endIndex - startIndex);
		}
	}

	return count;
}

// Calculate all slices
unsigned int ISubscale::calculateAllSlices(
	IDenseUnitCreator* denseUnitCreator,
	ISubspaceJoiner* subspaceJoiner,
	CsvDataHandler* csvHandler,
	ITableManager* tableManager,
	unsigned long long minSignature,
	unsigned long long maxSignature)
{
	unsigned long long deltaSig = (maxSignature - minSignature) / config->splittingFactor;
	unsigned long long minSigBoundary;
	unsigned long long maxSigBoundary;

	TimeMeasurement timer;
	timer.start();

	unsigned int numberOfEntries = 0;

	// iterate over all slices
	for (int j = 0; j < config->splittingFactor; j++)
	{
		// calculate signaturary boundaries of current slice
		int i = (j + config->splittingFactor / 2) % config->splittingFactor;

		minSigBoundary = minSignature + i * deltaSig;

		if (i < config->splittingFactor - 1)
		{
			maxSigBoundary = minSigBoundary + deltaSig;
		}
		else
		{
			maxSigBoundary = maxSignature + 1;
		}

		printf("%d. ----------------------------------------------\n", i + 1);


		// create dense units
		denseUnitCreator->createDenseUnits(minSigBoundary, maxSigBoundary);
		timer.createTimestamp(to_string(i) + ". Dense Units");

		// filter out all dense units that only appear in one dimension and copy dense units to a subspace table
		numberOfEntries = tableManager->duToSSTable(condensedSsTableWrapper->getPtr(), denseUnitTableWrapper->getPtr(), duTableSize);

		timer.createTimestamp(to_string(i) + ". Pruning");

		// check if slice is empty
		if (numberOfEntries > 0)
		{
			// join all entries by subspace
			subspaceJoiner->join(condensedSsTableWrapper->getPtr(), numberOfEntries);

			timer.createTimestamp(to_string(i) + ". Joining");

			// condense table
			numberOfEntries = tableManager->condenseTable(condensedSsTableWrapper->getPtr(), subspaceTableWrapper->getPtr(), ssTableSize);

			// copy table to local memory
			tableManager->deviceToLocal(localSubspaceTable, condensedSsTableWrapper->getPtr(), numberOfEntries);
			timer.createTimestamp(to_string(i) + ". Copying");
		}

		if (config->splittingFactor > 1)
		{
			// Write slice to a csv file
			std::string filePath = config->resultPath + std::to_string(i) + ".csv";
			csvHandler->writeTable(filePath.c_str(), localSubspaceTable, numberOfEntries);
			timer.createTimestamp(to_string(i) + ". Writing");

			// Clear dense unit and subspace tables
			denseUnitTableWrapper->reset();
			denseUnitCreator->clear();
			subspaceTableWrapper->reset();
			subspaceJoiner->clear();
			condensedSsTableWrapper->reset();
			timer.createTimestamp(to_string(i) + ". Resetting");
		}
	}

	std::string filePath = config->resultPath + "time_SliceCalculation.txt";
	timer.writeTimestampDeltas(filePath.c_str());

	return numberOfEntries;
}

// Combine all slices
unsigned int ISubscale::combineAllSlices(
	ISubspaceJoiner* subspaceJoiner,
	CsvDataHandler* csvHandler,
	ITableManager* tableManager
)
{
	unsigned int numberOfEntries;

	TimeMeasurement timer;
	timer.start();

	if (config->splittingFactor > 1)
	{
		for (int i = 0; i < config->splittingFactor; i++)
		{
			localSubspaceTable->reset();

			// read slice from file system
			std::string filePath = config->resultPath + std::to_string(i) + ".csv";
			numberOfEntries = csvHandler->readTable(filePath.c_str(), localSubspaceTable);

			// check if slice is empty
			if (numberOfEntries > 0)
			{
				// copy table to device
				tableManager->localToDevice(condensedSsTableWrapper->getPtr(), localSubspaceTable, numberOfEntries);
				timer.createTimestamp(to_string(i) + ". Load slice");

				// join all entries by subspace
				subspaceJoiner->join(condensedSsTableWrapper->getPtr(), numberOfEntries);
				timer.createTimestamp(to_string(i) + ". Join slice");
			}

		}
	}

	// filter out all entries with less ids than minPoints (only necessary when dense units are chosen to be smaller than the configured minPoints)
	// tableManager->filterIds(subspaceTableWrapper->getPtr(), ssTableSize, config->minPoints);

	// count number of candidates
	numberOfEntries = tableManager->countEntries(subspaceTableWrapper->getPtr(), ssTableSize);

	std::string filePath = config->resultPath + "time_SliceJoining.txt";
	timer.writeTimestampDeltas(filePath.c_str());

	return numberOfEntries;
}

LocalSubspaceTable* ISubscale::calculateClusterCandidates(vector<DataPoint> points)
{
	const int numberOfDimensions = points[0].values.size();
	const int numberOfPoints = points.size();

	// calculation classes
	LabelGenerator* labelGenerator = new LabelGenerator(1e14, 2e14);
	CoreSetCreator* coreSetCreator = new CoreSetCreator();
	CsvDataHandler* csvDataHandler = new CsvDataHandler();

	// variables
	unsigned long long* labels = new unsigned long long[numberOfPoints];
	unsigned long long minSignature;
	unsigned long long maxSignature;
	vector<vector<CoreSet>> coreSets;
	unsigned int numberOfEntries;


	//
	// generate labels
	labelGenerator->getLabels(labels, numberOfPoints);
	minSignature = labelGenerator->calcMinSignature(labels, points.size(), config->minPoints);
	maxSignature = labelGenerator->calcMaxSignature(labels, points.size(), config->minPoints);
	delete labelGenerator;

	//
	// generate core sets
	coreSets = coreSetCreator->createCoreSets(points, config->minPoints, config->epsilon);
	delete coreSetCreator;

	// count number of dense units that will be generated from core sets
	unsigned long long numberOfDenseUnits = countPossibleDenseUnits(coreSets, points.size(), config->minPoints);
	printf("Total number of dense units: %llu\n", numberOfDenseUnits);

	// calculate and combine slices to get the cluster candidates
	LocalSubspaceTable* resultTable = calculateCandidates(
		coreSets,
		csvDataHandler,
		labels,
		numberOfDimensions,
		numberOfPoints,
		minSignature,
		maxSignature
	);

	delete csvDataHandler;

	return resultTable;
}

