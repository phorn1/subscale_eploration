#pragma once

#include <string>

#include <nlohmann/json.hpp>

// Class to read and store config parameters
class SubscaleConfig
{
private:
    void fromJson(const nlohmann::json& j);


public:
    // configuation parameters
    int minPoints;
    double epsilon;
    int splittingFactor;
    std::string dataPath;
    std::string resultPath;
    bool useDBSCAN;
    bool saveCandidates;
    bool saveClusters;
    bool runSequential;
    int denseUnitTableSize;
    int subspaceTableSize;
    int finalTableSize; 
    int threadsPerBlock;
    int denseUnitsPerThread;

    void readJson(const char* path);
};

