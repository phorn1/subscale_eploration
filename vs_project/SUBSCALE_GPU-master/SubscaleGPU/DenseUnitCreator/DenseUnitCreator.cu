#include "DenseUnitCreator.cuh"
#include <math.h>
#include "DenseUnitKernels.cuh"


DenseUnitCreator::DenseUnitCreator(DenseUnitTable* denseUnitTable, int tableSize, int numberOfThreads, int denseUnitsPerThread)
{
    // Table Size has to be a prime number
    this->tableSize = tableSize;
    this->denseUnitTable = denseUnitTable;
    this->numberOfThreads = numberOfThreads;
    this->denseUnitsPerThread = denseUnitsPerThread;
}

DenseUnitCreator::~DenseUnitCreator()
{
    free();
}

// calculate all dense units in given dimension
void DenseUnitCreator::calculate(unsigned long long minSigBoundary, unsigned long long maxSigBoundary, int dimension)
{
    cudaError cudaStatus;

    for (int i = 0; i < coreSets[dimension].size(); i++)
    {
        // Calculate range of dense units that have to be created from the core set
        unsigned long long startIndex = local_BinomialCoeffCreator->choose(coreSets[dimension][i].pivot + 1, minPoints);
        unsigned long long endIndex = local_BinomialCoeffCreator->choose(coreSets[dimension][i].size, minPoints);

        // Launch Dense Unit Kernel
        int numberOfBlocks = ceil((double) (endIndex-startIndex) / (numberOfThreads*denseUnitsPerThread));

        denseUnitKernelMulti << <numberOfBlocks, numberOfThreads, sharedMemorySize, streams[i % numberOfStreams] >> > (
            dev_CoreSetIds[dimension][i], 
            coreSets[dimension][i].size, 
            dimension, 
            minPoints,
            startIndex, 
            endIndex, 
            minSigBoundary,
            maxSigBoundary,
            dev_labels,
            denseUnitMapWrapper->getPtr(), 
            binomialCoeffCreatorWrapper->getPtr(),
            denseUnitsPerThread);

        cudaStatus = cudaGetLastError();
        checkStatus(cudaStatus);
    }

    // Wait until all streams are finished before switching to the next dimension
    synchronizeKernelCall();
}

// calculate all dense units in all dimensions
void DenseUnitCreator::createDenseUnits(unsigned long long minSigBoundary, unsigned long long maxSigBoundary)
{
    for (int dim = 0; dim < coreSets.size(); dim++)
    {
        calculate(minSigBoundary, maxSigBoundary, dim);
    }
}

// initialze dense unit creator
void DenseUnitCreator::init(std::vector<std::vector<CoreSet>> coreSets, unsigned long long* labels, int numPoints, int minPoints)
{
    this->coreSets = coreSets;
    this->labels = labels;
    this->numPoints = numPoints;
    this->minPoints = minPoints;

    // Shared Memory for combinations and dense unit IDs
    sharedMemorySize = minPoints * numberOfThreads * sizeof(unsigned int) * 2;


    // Allocate required memory on device and host
    alloc();

    // Copy required host data to device
    copyToDevice();
}

// reset to initial state
void DenseUnitCreator::clear()
{
    denseUnitMapWrapper->reset();
}

// allocates memory for all objects on device and local host memory
void DenseUnitCreator::alloc()
{
    cudaError_t cudaStatus;

    // Streams
    streams = new cudaStream_t[numberOfStreams];
    for (int i = 0; i < numberOfStreams; i++)
    {
        cudaStatus = cudaStreamCreate(&streams[i]);
        checkStatus(cudaStatus);
    }

    // Core Sets
    dev_CoreSetIds = new unsigned int** [coreSets.size()];
    for (int dim = 0; dim < coreSets.size(); dim++)
    {
        dev_CoreSetIds[dim] = new unsigned int* [coreSets[dim].size()];

        for (int i = 0; i < coreSets[dim].size(); i++)
        {
            unsigned int* dev_ids;
            cudaStatus = cudaMalloc((void**)&dev_ids, coreSets[dim][i].size * sizeof(unsigned int));
            checkStatus(cudaStatus);
            dev_CoreSetIds[dim][i] = dev_ids;
        }
    }

    // Binomial Coefficients
    local_BinomialCoeffCreator = new LocalBinomialCoeffCreator(numPoints, minPoints);

    binomialCoeffCreatorWrapper = new DeviceBinomialCoeffCreator(numPoints, minPoints);

    // Dense Unit Map
    denseUnitMapWrapper = new DeviceDenseUnitMap(denseUnitTable, tableSize, 4);

    // Labels
    cudaStatus = cudaMalloc((void**)&dev_labels, numPoints * sizeof(unsigned long long));
    checkStatus(cudaStatus);
}

// copies labels and core sets to device
void DenseUnitCreator::copyToDevice()
{
    cudaError_t cudaStatus;

    int streamIndex = 0;

    // Labels
    cudaStatus = cudaMemcpyAsync(dev_labels, labels, numPoints * sizeof(unsigned long long), cudaMemcpyHostToDevice, streams[streamIndex++]);
    checkStatus(cudaStatus);

    // Core Sets
    for (int dim = 0; dim < coreSets.size(); dim++)
    {
        for (int i = 0; i < coreSets[dim].size(); i++)
        {
            cudaStatus = cudaMemcpyAsync(dev_CoreSetIds[dim][i], coreSets[dim][i].ids, coreSets[dim][i].size * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[i % numberOfStreams]);
            checkStatus(cudaStatus);
        }
    }

    cudaStatus = cudaDeviceSynchronize();
    checkStatus(cudaStatus);
}

// frees memory of all objects
void DenseUnitCreator::free()
{
    cudaError_t cudaStatus;
    // Core Sets
    for (int dim = 0; dim < coreSets.size(); dim++)
    {
        for (int i = 0; i < coreSets[dim].size(); i++)
        {
            cudaStatus = cudaFree(dev_CoreSetIds[dim][i]);
            checkStatus(cudaStatus);
        }

        delete[] dev_CoreSetIds[dim];
    }
    delete[] dev_CoreSetIds;

    // Labels
    cudaStatus = cudaFree(dev_labels);
    checkStatus(cudaStatus);

    // Streams
    for (int i = 0; i < numberOfStreams; i++)
    {
        cudaStatus = cudaStreamDestroy(streams[i]);
        checkStatus(cudaStatus);
    }
    delete[] streams;

    // Binomial Coefficients
    delete local_BinomialCoeffCreator;
    delete binomialCoeffCreatorWrapper;

    // Collision Map
    delete denseUnitMapWrapper;
}

