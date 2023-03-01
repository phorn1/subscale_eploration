#pragma once
#include <vector>
#include <list>

// Data point
struct DataPoint {
	unsigned int id;
	std::vector<double> values;
};

// Core Set
struct CoreSet {
	unsigned int* ids;
	int pivot;
	int size;
};

// Struct for sorting values in one dimension (see CoreSetCreator)
struct PointIDValue {
	unsigned int id;
	double value;

	bool operator < (const PointIDValue& str) const
	{
		return (value < str.value);
	}
};

// Cluster
struct Cluster {
	std::vector<unsigned int> subspace;
	std::vector<unsigned int> ids;
};
