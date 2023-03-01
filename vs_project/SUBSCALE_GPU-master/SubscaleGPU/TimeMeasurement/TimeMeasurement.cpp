#include "TimeMeasurement.h"
#include <fstream>

// Starts timer
void TimeMeasurement::start()
{
    startTime = chrono::steady_clock::now();
}

// Add a new timestamp
void TimeMeasurement::createTimestamp(string title)
{
    chrono::steady_clock::time_point currentTime = chrono::steady_clock::now();

    long delta = chrono::duration_cast<chrono::milliseconds>(currentTime - startTime).count();

    timestamps.push_back(pair<string, long>(title, delta));
}

// Prints the time differences between the timestamps
void TimeMeasurement::printTimestampDeltas()
{
    long previousTime = 0;

    for (pair<string, long> timestamp : timestamps)
    {
        long delta = timestamp.second - previousTime;

        printf("%s: %d\n", timestamp.first.c_str(), delta);

        previousTime = timestamp.second;
    }

    printf("\nFull runtime: %d\n", (timestamps.end() - 1)->second);
}

// Write the time differences between the timestamps to a file
void TimeMeasurement::writeTimestampDeltas(const char* path)
{
    std::ofstream myFile(path);

    long previousTime = 0;

    for (pair<string, long> timestamp : timestamps)
    {
        long delta = timestamp.second - previousTime;

        myFile << timestamp.first + ": " + std::to_string(delta) + " ms\n";

        previousTime = timestamp.second;
    }

    myFile << "\nFull runtime: " + std::to_string((timestamps.end() - 1)->second) + " ms\n";
}
