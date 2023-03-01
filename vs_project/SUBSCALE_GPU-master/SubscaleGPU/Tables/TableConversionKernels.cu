#include "TableConversionKernels.cuh"

#include <cstdint>

// removes all entries with less dimensions in the subspace than the given number
__global__ void tableDimensionsFilterKernel(DenseUnitTable* table, int minNumberOfDims)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    if (x < table->getTableSize())
    {
        uint32_t* dimensions = table->getDimensions(x);
        int dimensionsSize = table->getDimensionsSize();

        int count = 0;

        for (int i = 0; i < dimensionsSize; i++)
        {
            count += __popc(dimensions[i]); // __popc counts number of set bits in a 32 bit integer
        }

        // check if the bucket isn't empty and if the number of dimensions in the subspace are smaller than minNumberOfDims
        if (count > 0 && count < minNumberOfDims)
        {
            // remove entry from table
            table->setIdsZero(x);
            table->setDimensionsZero(x);
        }
    }
}

// removes all entries with less added ids than the given number
__global__ void tableIdsFilterKernel(SubspaceTable* table, int minNumberOfIds)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    if (x < table->getTableSize())
    {
        uint32_t* ids = table->getIds(x);
        int idsSize = table->getIdsSize();

        int count = 0;

        for (int i = 0; i < idsSize; i++)
        {
            count += __popc(ids[i]); // __popc counts number of set bits in a 32 bit integer
        }

        // check if the bucket isn't empty and if the number of added ids is smaller than minNumberOfIds 
        if (count > 0 && count < minNumberOfIds)
        {
            // remove entry from table
            table->setIdsZero(x);
            table->setDimensionsZero(x);
        }
    }
}

// searches source table for all non-empty entries and inserts them into target table in sequential order
__global__ void tableCondenseKernel(SubscaleTable* targetTable, SubscaleTable* sourceTable, unsigned int* index)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    if (x < sourceTable->getTableSize())
    {
        uint32_t* ids = sourceTable->getIds(x);
        int idsSize = sourceTable->getIdsSize();
        uint32_t* dimensions = sourceTable->getDimensions(x);
        int dimensionsSize = sourceTable->getDimensionsSize();

        uint32_t* entry;
        int entrySize;

        if (idsSize < dimensionsSize)
        {
            entry = ids;
            entrySize = idsSize;
        }
        else
        {
            entry = dimensions;
            entrySize = dimensionsSize;
        }

        for (int i = 0; i < entrySize; i++)
        {
            if (entry[i] != 0)
            {
                unsigned int targetIndex = atomicAdd(index, 1);

                targetTable->insertIds(ids, targetIndex);
                targetTable->insertDimensions(dimensions, targetIndex);
                break;
            }
        }
    }
}

// copies data from a condensed dense unit table to a condensed subspace table
__global__ void tableCondenseKernel(SubspaceTable* targetTable, DenseUnitTable* sourceTable, unsigned int* index)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    if (x < sourceTable->getTableSize())
    {
        uint32_t* ids = sourceTable->getIds(x);
        int idsSize = sourceTable->getIdsSize();
        uint32_t* dimensions = sourceTable->getDimensions(x);
        int dimensionsSize = sourceTable->getDimensionsSize();

        uint32_t* entry;
        int entrySize;

        if (idsSize < dimensionsSize)
        {
            entry = ids;
            entrySize = idsSize;
        }
        else
        {
            entry = dimensions;
            entrySize = dimensionsSize;
        }

        for (int i = 0; i < entrySize; i++)
        {
            if (entry[i] != 0)
            {
                unsigned int targetIndex = atomicAdd(index, 1);

                targetTable->dev_addIds(ids, idsSize, targetIndex);
                targetTable->insertDimensions(dimensions, targetIndex);
                break;
            }
        }
    }
}

__global__ void tableCountKernel(SubscaleTable* table, unsigned int* count)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    if (x < table->getTableSize())
    {
        uint32_t* ids = table->getIds(x);
        int idsSize = table->getIdsSize();
        uint32_t* dimensions = table->getDimensions(x);
        int dimensionsSize = table->getDimensionsSize();

        uint32_t* entry;
        int entrySize;

        if (idsSize < dimensionsSize)
        {
            entry = ids;
            entrySize = idsSize;
        }
        else
        {
            entry = dimensions;
            entrySize = dimensionsSize;
        }

        for (int i = 0; i < entrySize; i++)
        {
            if (entry[i] != 0)
            {
                unsigned int targetIndex = atomicAdd(count, 1);
                break;
            }
        }
    }
}

/*
// converts a dense unit table to a subspace table 
__global__ void denseUnitToSubspaceKernel(SubspaceTable* targetTable, DenseUnitTable* sourceTable, unsigned int* index)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    if (x < sourceTable->getTableSize())
    {
        uint32_t* ids = sourceTable->getIds(x);
        int idsSize = sourceTable->getIdsSize();

        // Check if table bucket isn't empty
        if (ids[1] != 0)
        {
            uint32_t* dimensions = sourceTable->getDimensions(x);
            int dimensionsSize = sourceTable->getDimensionsSize();

            int dimCount = 0;

            // count the number of dimension elements that are not zero
            for (int j = 0; j < dimensionsSize; j++)
            {
                if (dimensions[j] != 0)
                {
                    dimCount++;

                    // if only one dimension element is not zero, the number of set bits have to be checked
                    if (dimCount == 1)
                    {
                        // check if more than 1 Bit is set (see Brian Kernighan)
                        if (dimensions[j] & (dimensions[j] - 1))
                        {
                            // increase table index by one with an atomic operation
                            unsigned int myIndex = atomicAdd(index, 1);

                            // add ids to subspace table
                            targetTable->dev_addIds(ids, idsSize, myIndex);

                            // insert dimensions into subspace table
                            targetTable->insertDimensions(dimensions, myIndex);
                            break;
                        }
                    }
                    else if (dimCount > 1)
                    {
                        // increase table index by one with an atomic operation
                        unsigned int myIndex = atomicAdd(index, 1);

                        // add ids to subspace table
                        targetTable->dev_addIds(ids, idsSize, myIndex);

                        // insert dimensions into subspace table
                        targetTable->insertDimensions(dimensions, myIndex);
                        break;
                    }
                }
            }
        }
    }
}

__global__ void subspaceToSubspaceKernel(SubspaceTable* targetTable, SubspaceTable* sourceTable, int minIds, unsigned int* index)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    if (x < sourceTable->getTableSize())
    {
        unsigned int* ids = sourceTable->getIds(x);

        int idsSize = sourceTable->getIdsSize();

        int count = 0;

        for (int i = 0; i < idsSize; i++)
        {
            count += __popc(ids[i]); // __popc counts number of set bits in a 32 bit integer
        }


        if (count >= minIds)
        {
            unsigned int* dimensions = sourceTable->getDimensions(x);

            // increase table index by one with an atomic operation
            unsigned int myIndex = atomicAdd(index, 1);

            // insert ids and dimensions into the target subspace table
            targetTable->insertIds(ids, myIndex);
            targetTable->insertDimensions(dimensions, myIndex);
        }
    }
}
*/