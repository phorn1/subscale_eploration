#pragma once

#include "../Tables/LocalSubspaceTable.cuh"
#include "../SubscaleTypes.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/dbscan/dbscan.hpp"
#include <vector>
#include <thread>

class MultithreadedClustering
{
private:
	int minPoints;
	double epsilon;

	void calculateSomeClusters(vector<DataPoint> points, LocalSubspaceTable* clusterCandidates, int startIndex, int endIndex, vector<Cluster> result);

public:
	MultithreadedClustering(int minPoints, double epsilon);

	std::vector<Cluster> calculateClusters(vector<DataPoint> points, LocalSubspaceTable* clusterCandidates);
};

