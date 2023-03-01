#include "SubspaceJoiner.cuh"
#include "device_launch_parameters.h"
#include <math.h>

// CUDA kernel to join entries by their subspace
__global__ void joinKernel(SubspaceMap* subspaceMap, SubspaceTable* sourceTable, unsigned int tableSize)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id < tableSize)
    {
        unsigned int* ids = sourceTable->getIds(id);
        unsigned int* dimensions = sourceTable->getDimensions(id);
        subspaceMap->insertEntry(ids, dimensions);
    }
}

// Constructor
SubspaceJoiner::SubspaceJoiner(SubspaceTable* subspaceTable, int tableSize, int numberOfThreads)
{
    this->tableSize = tableSize;
    this->subspaceTable = subspaceTable;
    this->numberOfThreads = numberOfThreads;
}

// Initializes hash map
void SubspaceJoiner::init(unsigned int ticketSize)
{
    subspaceMapWrapper = new DeviceSubspaceMap(subspaceTable, tableSize, ticketSize);
}

// Deletes hash map
void SubspaceJoiner::free()
{
    delete subspaceMapWrapper;
}

// Clears hash map
void SubspaceJoiner::clear()
{
    subspaceMapWrapper->reset();
}

// Inserts all entries from a table into the hash map and thereby joins them to cluster candidates
void SubspaceJoiner::join(SubspaceTable* sourceTable, int ssTableSize)
{
    int numberOfBlocks = ceil((double)ssTableSize / numberOfThreads);
    joinKernel << <numberOfBlocks, numberOfThreads >> > (subspaceMapWrapper->getPtr(), sourceTable, ssTableSize);
    synchronizeKernelCall();
}
