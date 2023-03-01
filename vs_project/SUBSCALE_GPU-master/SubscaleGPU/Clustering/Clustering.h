#pragma once
#include "../Tables/LocalSubspaceTable.cuh"
#include "../SubscaleTypes.h"

// class for final clustering
class Clustering
{
private:
	int minPoints;
	double epsilon;

public:
	Clustering(int minPoints, double epsilon);

	std::vector<Cluster> calculateClusters(vector<DataPoint> points, LocalSubspaceTable* clusterCandidates, vector<vector<double>> &scoring, vector<vector<double>> &optEps);
	std::vector<Cluster> calculateClusters(vector<DataPoint> points, LocalSubspaceTable* clusterCandidates, vector<int> minPointVector, vector<double> epsilonVector);
};

