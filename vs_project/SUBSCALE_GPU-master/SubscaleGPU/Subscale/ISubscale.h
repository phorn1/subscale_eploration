#pragma once

#include "../BinomialCoeffCreator/LocalBinomialCoeffCreator.cuh"
#include "../CoreSetCreator/CoreSetCreator.h"
#include "../LabelGenerator/LabelGenerator.h"
#include "../Tables/LocalSubspaceTable.cuh"
#include "../Tables/ITableManager.h"
#include "../SubspaceJoiner/ISubspaceJoiner.h"
#include "../DenseUnitCreator/IDenseUnitCreator.h"
#include "../CsvDataHandler/CsvDataHandler.h"
#include "../SubscaleConfig/SubscaleConfig.h"
#include "../TimeMeasurement/TimeMeasurement.h"



// Interface for the subscale implementations
class ISubscale
{
protected:

    SubscaleConfig* config;

    int duTableSize;
    int ssTableSize;
    int condensedSsTableSize;


    MemoryManager<DenseUnitTable>* denseUnitTableWrapper;
    MemoryManager<SubspaceTable>* subspaceTableWrapper;
    MemoryManager<SubspaceTable>* condensedSsTableWrapper;
    LocalSubspaceTable* localSubspaceTable;

    virtual LocalSubspaceTable* calculateCandidates(
        vector<vector<CoreSet>> coreSets,
        CsvDataHandler* csvHandler,
        unsigned long long* labels,
        int numberOfDimensions,
        int numberOfPoints,
        unsigned long long minSignature,
        unsigned long long maxSignature) = 0;

    unsigned long long countPossibleDenseUnits(
        vector<vector<CoreSet>> coreSets,
        int numPoints,
        int minPoints);

    unsigned int calculateAllSlices(
        IDenseUnitCreator* denseUnitCreator,
        ISubspaceJoiner* subspaceJoiner,
        CsvDataHandler* csvHandler,
        ITableManager* tableManager,
        unsigned long long minSignature,
        unsigned long long maxSignature
    );

    unsigned int combineAllSlices(
        ISubspaceJoiner* subspaceJoiner,
        CsvDataHandler* csvHandler,
        ITableManager* tableManager
    );


public:
    ISubscale(SubscaleConfig* config)
    {
        this->config = config;

        this->duTableSize = roundToNextPrime(config->denseUnitTableSize);
        this->ssTableSize = roundToNextPrime(config->subspaceTableSize);
        this->condensedSsTableSize = roundToNextPrime(config->subspaceTableSize);
    }

    LocalSubspaceTable* calculateClusterCandidates(vector<DataPoint> points);
};