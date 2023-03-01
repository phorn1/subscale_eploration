#pragma once

#include "../Tables/DeviceDenseUnitTable.cuh"
#include "../Tables/DeviceSubspaceTable.cuh"
#include "../Tables/LocalSubspaceTable.cuh"
#include "../Tables/TableManager.cuh"
#include "../DenseUnitCreator/DenseUnitCreator.cuh"
#include "../SubspaceJoiner/SubspaceJoiner.cuh"
#include "../CsvDataHandler/CsvDataHandler.h"
#include "../SubscaleConfig/SubscaleConfig.h"
#include "ISubscale.h"



// Parallel subscale implementation
class Subscale : public ISubscale
{
private:

    LocalSubspaceTable* calculateCandidates(
        vector<vector<CoreSet>> coreSets, 
        CsvDataHandler* csvHandler,
        unsigned long long* labels,
        int numberOfDimensions,
        int numberOfPoints,
        unsigned long long minSignature, 
        unsigned long long maxSignature);


public:
    Subscale(SubscaleConfig* config) : ISubscale(config)
    {}
};

