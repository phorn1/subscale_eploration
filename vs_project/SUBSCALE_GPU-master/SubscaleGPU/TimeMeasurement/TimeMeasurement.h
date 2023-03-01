#pragma once

#include <chrono>
#include <vector>
#include <string>

using namespace std;

// Class to track execution times
class TimeMeasurement
{
private:
	chrono::steady_clock::time_point startTime;

	vector<pair<string, long>> timestamps;

public:
	void start();

	void createTimestamp(string title);

	void printTimestampDeltas();

	void writeTimestampDeltas(const char* path);
};

