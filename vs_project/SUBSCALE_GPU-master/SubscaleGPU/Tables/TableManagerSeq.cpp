#include "TableManagerSeq.h"
#include <stdexcept>
#include <cstring>

int TableManagerSeq::countBits(unsigned int value)
{
    int count = 0;

    while (value)
    {
        count++;
        value = value & (value - 1);
    }

    return count;
}

void TableManagerSeq::deviceToLocal(SubscaleTable* localTable, SubscaleTable* deviceTable, unsigned int numberOfEntries)
{
	// Empty because only sequential local to local operations are supported
}

void TableManagerSeq::localToDevice(SubscaleTable* deviceTable, SubscaleTable* localTable, unsigned int numberOfEntries)
{
	// Empty because only sequential local to local operations are supported
}

void TableManagerSeq::localToLocal(SubscaleTable* targetTable, SubscaleTable* sourceTable, unsigned int numberOfEntries)
{
    int idsSize = sourceTable->getIdsSize();
    int dimensionsSize = sourceTable->getDimensionsSize();
    unsigned int* ids = sourceTable->getIds();
    unsigned int* dimensions = sourceTable->getDimensions();

    if (idsSize != targetTable->getIdsSize() ||
        dimensionsSize != targetTable->getDimensionsSize())
    {
        throw std::runtime_error("Table manager error: localToLocal() requires tables of the same size!");
    }

    // Copy ids
    std::memcpy(targetTable->getIds(), ids, sizeof(unsigned int) * idsSize * numberOfEntries);

    // Copy dimensions
    std::memcpy(targetTable->getDimensions(), dimensions, sizeof(unsigned int) * dimensionsSize * numberOfEntries);
}

// filter out all entries with fewer IDs than minNumberOfIds
void TableManagerSeq::filterIds(SubspaceTable* table, int tableSize, int minNumberOfIds)
{
	for (int x = 0; x < tableSize; x++)
	{
        uint32_t* ids = table->getIds(x);
        int idsSize = table->getIdsSize();

        int count = 0;

        for (int i = 0; i < idsSize; i++)
        {
            count += countBits(ids[i]);
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

// filter out all entries with fewer dimensions than minNumberOfDimensions
void TableManagerSeq::filterDimensions(DenseUnitTable* table, int tableSize, int minNumberOfDimensions)
{
    for (int x = 0; x < tableSize; x++)
    {
        uint32_t* dimensions = table->getDimensions(x);
        int dimensionsSize = table->getDimensionsSize();

        int count = 0;

        for (int i = 0; i < dimensionsSize; i++)
        {
            count += countBits(dimensions[i]);
        }

        // check if the bucket isn't empty and if the number of added ids is smaller than minNumberOfIds 
        if (count > 0 && count < minNumberOfDimensions)
        {
            // remove entry from table
            table->setIdsZero(x);
            table->setDimensionsZero(x);
        }
    }
}

// copy entries of tables with the same type
unsigned int TableManagerSeq::condenseTable(SubscaleTable* targetTable, SubscaleTable* sourceTable, int sourceTableSize)
{
    unsigned int targetIndex = 0;

    for (int x = 0; x < sourceTableSize; x++)
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
                targetTable->insertIds(ids, targetIndex);
                targetTable->insertDimensions(dimensions, targetIndex);
                targetIndex++;
                break;
            }
        }
    }


	return targetIndex;
}

unsigned int TableManagerSeq::copyDenseUnitToSubspaceTable(SubspaceTable* targetTable, DenseUnitTable* sourceTable, int sourceTableSize)
{
    unsigned int targetIndex = 0;

    for (int x = 0; x < sourceTableSize; x++)
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
                targetTable->addIds(ids, idsSize, targetIndex);
                targetTable->insertDimensions(dimensions, targetIndex);
                targetIndex++;
                break;
            }
        }
    }


    return targetIndex;
}

// filter and copy dense units with more than 1 dimension to a subspace table
unsigned int TableManagerSeq::duToSSTable(SubspaceTable* targetTable, DenseUnitTable* sourceTable, int sourceTableSize)
{
    int idsSize = sourceTable->getIdsSize();
    int dimensionsSize = sourceTable->getDimensionsSize();
    unsigned int targetIndex = 0;

    for (int i = 0; i < sourceTableSize; i++)
    {
        unsigned int* ids = sourceTable->getIds(i);
        if (ids[0] != ids[1])
        {
            unsigned int* dimensions = sourceTable->getDimensions(i);
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
                            targetTable->addIds(ids, idsSize, targetIndex);
                            targetTable->insertDimensions(dimensions, targetIndex);
                            targetIndex++;
                            break;
                        }
                    }
                    else if (dimCount > 1)
                    {
                        targetTable->addIds(ids, idsSize, targetIndex);
                        targetTable->insertDimensions(dimensions, targetIndex);
                        targetIndex++;
                        break;
                    }
                }

            }
        }
    }

    return targetIndex;
}

unsigned int TableManagerSeq::countEntries(SubscaleTable* table, int tableSize)
{
    unsigned int count = 0;

    for (int x = 0; x < tableSize; x++)
    {
        int idsSize = table->getIdsSize();
        int dimensionsSize = table->getDimensionsSize();

        uint32_t* entry;
        int entrySize;

        if (idsSize < dimensionsSize)
        {
            entry = table->getIds(x);;
            entrySize = idsSize;
        }
        else
        {
            entry = table->getDimensions(x);;
            entrySize = dimensionsSize;
        }

        for (int i = 0; i < entrySize; i++)
        {
            if (entry[i] != 0)
            {
                count++;
                break;
            }
        }
    }

    return count;
}
