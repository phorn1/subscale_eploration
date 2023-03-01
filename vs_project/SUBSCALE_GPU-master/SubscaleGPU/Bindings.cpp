#include <vector>
#include <string>

#include "CsvDataHandler/CsvDataHandler.h"
#include "Subscale/Subscale.h"
#include "Clustering/Clustering.h"

#ifdef _WINDLL
#define SUBSCALE_API __declspec(dllexport)
#else
#define SUBSCALE_API __declspec(dllimport)
#endif

struct DataPointBinding {
    unsigned int id;
    unsigned int nValues;
    double* values;
};

struct Subspace {
    unsigned int* dimensions;
    unsigned int* ids;
    unsigned int nDimensions;
    unsigned int nIds;
};

using namespace std;

//read file
extern "C" SUBSCALE_API int readData(DataPointBinding** points, string filepath, char delimiter)
{
    CsvDataHandler* csvHandler = new CsvDataHandler();

    vector<DataPoint> readPoints;
    readPoints = csvHandler->read(filepath.c_str(), delimiter);

    DataPointBinding* hep = (DataPointBinding*)malloc(sizeof(DataPointBinding) * readPoints.size());

    int i = 0;

    for (DataPoint dp : readPoints)
    {
        hep[i].id = dp.id;

        hep[i].values = dp.values.data();
        hep[i].nValues = dp.values.size();

        i++;
    }

    //*points = hep;

    return readPoints.size();
}


extern "C" SUBSCALE_API void freeData(DataPointBinding* points, int nPoints)
{
    
}


extern "C" SUBSCALE_API void executeSubscale(DataPointBinding * points, LocalSubspaceTable** pClusterCandidates, unsigned int nPoints, struct Subspace** subspaces, unsigned int* nSubspaces, double eps, int minP)
{
    ISubscale* subscale;

    SubscaleConfig* config = new SubscaleConfig();

    config->denseUnitsPerThread = 3;
    config->denseUnitTableSize = 8000000;
    config->epsilon = eps;
    config->finalTableSize = 3000000;
    config->minPoints = minP;
    config->resultPath = "results/";
    config->splittingFactor = 16;
    config->subspaceTableSize = 5000000;
    config->threadsPerBlock = 64;

    vector<DataPoint> dataPoints;

    for (int i = 0; i < nPoints; i++)
    {
        DataPoint p;
        p.id = points[i].id;
        
        vector<double> v;

        copy(&points[i].values[0], &points[i].values[points[i].nValues], back_inserter(v));

        p.values = v;

        dataPoints.push_back(p);
    }

    subscale = new Subscale(config);

    LocalSubspaceTable* clusterCandidates = subscale->calculateClusterCandidates(dataPoints);

    struct Subspace* subspaceTable = (struct Subspace*)malloc(sizeof(struct Subspace*) * clusterCandidates->getTableSize());

    for (int i = 0; i < clusterCandidates->getTableSize(); i++)
    {
        subspaceTable[i].dimensions = clusterCandidates->getDimensionsVec(i).data();
        subspaceTable[i].ids = clusterCandidates->getIdsVec(i).data();
        subspaceTable[i].nDimensions = clusterCandidates->getDimensionsVec(i).size();
        subspaceTable[i].nIds = clusterCandidates->getIdsVec(i).size();
    }

    *pClusterCandidates = clusterCandidates;

    *subspaces = subspaceTable;
    *nSubspaces = clusterCandidates->getTableSize();
}

//execute subscale
//ISubscale* subscale;
//
//if (config->runSequential)
//{
//    // sequential
//    subscale = new SubscaleSeq(config);
//}
//else
//{
//    // parallel
//    subscale = new Subscale(config);
//}
//
//LocalSubspaceTable* clusterCandidates = subscale->calculateClusterCandidates(points);

extern "C" SUBSCALE_API void executeDBScan(DataPointBinding * points, LocalSubspaceTable * clusterCandidates, Subspace ** clusterTableRet, unsigned int* nClusters, unsigned int nPoints, double eps, int minP)
{
    vector<DataPoint> dataPoints;

    for (int i = 0; i < nPoints; i++)
    {
        DataPoint p;
        p.id = points[i].id;

        vector<double> v;

        copy(&points[i].values[0], &points[i].values[points[i].nValues], back_inserter(v));

        p.values = v;

        dataPoints.push_back(p);
    }

    struct Subspace* subspaceTable = (struct Subspace*)malloc(sizeof(struct Subspace*) * clusterCandidates->getTableSize());

    for (int i = 0; i < clusterCandidates->getTableSize(); i++)
    {
        subspaceTable[i].dimensions = clusterCandidates->getDimensionsVec(i).data();
        subspaceTable[i].ids = clusterCandidates->getIdsVec(i).data();
        subspaceTable[i].nDimensions = clusterCandidates->getDimensionsVec(i).size();
        subspaceTable[i].nIds = clusterCandidates->getIdsVec(i).size();
    }

    Clustering* finalClustering = new Clustering(minP, eps);
    std::vector<Cluster> clusters = finalClustering->calculateClusters(dataPoints, clusterCandidates);

    struct Subspace* clusterTable = (struct Subspace*)malloc(sizeof(struct Subspace*) * clusters.size());

    for (int i = 0; i < clusters.size(); i++)
    {
        clusterTable[i].dimensions = clusters[i].subspace.data();
        clusterTable[i].ids = clusters[i].ids.data();
        clusterTable[i].nDimensions = clusters[i].subspace.size();
        clusterTable[i].nIds = clusters[i].ids.size();
    }

    *clusterTableRet = clusterTable;
    *nClusters = clusters.size();
}

//execute db scan
// Search cluster candidates for real clusters with the DBSCAN algorithm
//Clustering* finalClustering = new Clustering(config->minPoints, config->epsilon);
//std::vector<Cluster> clusters = finalClustering->calculateClusters(points, clusterCandidates);


//TODO
//execute db scan (minPoints, epsilon param)
// Search cluster candidates for real clusters with the DBSCAN algorithm
//Clustering* finalClustering = new Clustering(config->minPoints, config->epsilon);
//std::vector<Cluster> clusters = finalClustering->calculateClusters(points, clusterCandidates, minPointVector, epsilonVector);
