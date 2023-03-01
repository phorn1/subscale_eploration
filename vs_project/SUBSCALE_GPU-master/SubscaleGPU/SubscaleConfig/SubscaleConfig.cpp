#include "SubscaleConfig.h"
#include <fstream>

// extracts values from a json object and stores them in the member variables
void SubscaleConfig::fromJson(const nlohmann::json& j)
{
    j.at("minPoints").get_to(minPoints);
    j.at("epsilon").get_to(epsilon);
    j.at("splittingFactor").get_to(splittingFactor);
    j.at("dataPath").get_to(dataPath);
    j.at("resultPath").get_to(resultPath);
    j.at("useDBSCAN").get_to(useDBSCAN);
    j.at("saveCandidates").get_to(saveCandidates);
    j.at("saveClusters").get_to(saveClusters);
    j.at("useDBSCAN").get_to(useDBSCAN);
    j.at("runSequential").get_to(runSequential);
    j.at("denseUnitTableSize").get_to(denseUnitTableSize);
    j.at("subspaceTableSize").get_to(subspaceTableSize);
    j.at("finalTableSize").get_to(finalTableSize);
    j.at("threadsPerBlock").get_to(threadsPerBlock);
    j.at("denseUnitsPerThread").get_to(denseUnitsPerThread);
}

// reads a json file and stores the values in the member variables
void SubscaleConfig::readJson(const char* path)
{
    // Create an input filestream
    std::ifstream configFile(path);

    // Make sure the file is open
    if (!configFile.is_open()) throw std::runtime_error("Could not open config file");

    std::string line;
    std::string text;

    while (getline(configFile, line))
    {
        text.append(line);
    }

    // Convert string to json object
    nlohmann::json j = nlohmann::json::parse(text);

    // Read values from json object
    fromJson(j);
}
