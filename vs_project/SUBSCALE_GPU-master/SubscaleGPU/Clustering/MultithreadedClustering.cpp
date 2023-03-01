#include "MultithreadedClustering.h"

using namespace mlpack;
using namespace std;

MultithreadedClustering::MultithreadedClustering(int minPoints, double epsilon)
{
	this->minPoints = minPoints;
	this->epsilon = epsilon;
}

// find clusters in the given cluster candidates
vector<Cluster> MultithreadedClustering::calculateClusters(vector<DataPoint> points, LocalSubspaceTable* clusterCandidates)
{
	vector<Cluster> clusters;

	int numThreads = std::thread::hardware_concurrency();

	int candidatesPerThread = clusterCandidates->getTableSize() / numThreads;
	int excessCandidates = clusterCandidates->getTableSize() % numThreads;

	int startIndex = 0;

	vector<vector<Cluster>> results;

	//create threads
	vector<std::thread> workers;
	for (int i = 0; i < numThreads; i++)
	{
		results.push_back(vector<Cluster>());

		int candidates = candidatesPerThread;
		if (excessCandidates > 0)
		{
			candidates++;
			excessCandidates--;
		}

		vector<Cluster> result = results[i];

		std::thread worker(&MultithreadedClustering::calculateSomeClusters, this, points, clusterCandidates, startIndex, startIndex + candidates, result);
		workers.push_back(worker);

		startIndex += candidates;
	}

	//join threads
	for (int i = 0; i < numThreads; i++)
	{
		workers[i].join();
		clusters.insert(clusters.end(), results[i].begin(), results[i].end());
	}

	return clusters;
}

void MultithreadedClustering::calculateSomeClusters(vector<DataPoint> points, LocalSubspaceTable* clusterCandidates, int startIndex, int endIndex, vector<Cluster> result)
{
	dbscan::DBSCAN<> dbs(epsilon, minPoints);
	
	for (int i = startIndex; i < endIndex; i++)
	{
		arma::mat data;

		//
		// convert entry to a matrix
		vector<unsigned int> ids = clusterCandidates->getIdsVec(i);
		vector<unsigned int> dimensions = clusterCandidates->getDimensionsVec(i);

		data.set_size(dimensions.size(), ids.size());

		for (int j = 0; j < ids.size(); j++)
		{
			for (int k = 0; k < dimensions.size(); k++)
			{
				data(k, j) = points[ids[j]].values[dimensions[k]];
			}
		}

		//
		// clustering
		arma::mat centroids;
		arma::Row<size_t> assignements;

		int numClusters = dbs.Cluster(data, assignements, centroids);

		// check number of found clusters
		if (numClusters > 0)
		{
			//
			// convert cluster to a Cluster struct
			vector<vector<unsigned int>> pointsInClusters;
			pointsInClusters.resize(numClusters);

			for (int j = 0; j < ids.size(); j++)
			{
				if (assignements[j] != -1)
				{
					pointsInClusters[assignements[j]].push_back(ids[j]);
				}
			}

			for (std::vector<unsigned int> pointsInCluster : pointsInClusters)
			{
				Cluster cluster = { dimensions, pointsInCluster };
				result.push_back(cluster);
			}
		}
	}
}
