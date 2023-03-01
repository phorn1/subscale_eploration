#pragma once

#include <vector>
#include "../SubscaleTypes.h"
#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <algorithm>

#include "../Tables/SubspaceTable.cuh"

using namespace std;

// class for all IO operations
class CsvDataHandler
{
private:
	const unsigned long long bufferSize = 1000000000; //50000000; // 50 MB - If larger files have to be read and written, bufferSize has to be increased
	char* buffer = new char[bufferSize];

public:
	vector<DataPoint> read(const char* path, char delimiter);
	void writeTable(const char* path, SubscaleTable* table, unsigned int numberOfEntries);
	void writeVecTable(const char* path, SubspaceTable* table, unsigned int numberOfEntries);

	void writeClusters(const char* path, vector<Cluster> clusters);

	unsigned int readTable(const char* path, SubspaceTable* table);
};

