#include "DenseUnitCreatorSeq.h"
#include "cuda_runtime.h"

// calculates all dense units from given core sets
void DenseUnitCreatorSeq::denseUnitCalc(
    const unsigned int* coreSetIds, 
    const int coreSetSize, 
    int pivot,
    const int dimension,
    const unsigned long long minSigBoundary,
    const unsigned long long maxSigBoundary)
{
    unsigned long long signature;
    signatureStack[0] = 0;

    // initialize with MINPOINT elements, starting from the last element int the array
    for (int i = 0; i < minPoints; i++)
    {
        indexStack[i] = coreSetSize - 1 - i;

        denseUnitIds[i] = coreSetIds[indexStack[i]];

        signatureStack[i + 1] = signatureStack[i] + labels[denseUnitIds[i]];
    }

    // Add first Dense Unit
    signature = signatureStack[minPoints];

    if (signature >= minSigBoundary && signature < maxSigBoundary)
    {
        denseUnitMap->insertDenseUnit(signature, denseUnitIds, dimension);
    }
    

    // if the pivot index is too small, adjust it so that all possible combinations are calculated
    if (pivot < minPoints - 2)
    {
        pivot = minPoints - 2;
    }

    // calculate combinations until the last combination starting with the pivot index is reached
    while (indexStack[0] > pivot + 1 || indexStack[1] > minPoints - 2)
    {
        // starting from the last index, search for the first index that can be decremented further
        int levelDepth = minPoints - 1;
        while ((indexStack[levelDepth] == (minPoints - 1 - levelDepth)) && levelDepth != 0)
        {
            levelDepth--;
        }

        // decrement index
        indexStack[levelDepth]--;

        // add ID and calculate signature with the new index of the current level

        denseUnitIds[levelDepth] = coreSetIds[indexStack[levelDepth]];


        signatureStack[levelDepth + 1] = signatureStack[levelDepth] + labels[denseUnitIds[levelDepth]];


        // iterate through the following levels until the last level is reached
        while (levelDepth < minPoints - 1)
        {
            levelDepth++;

            // decrement the index with each level
            indexStack[levelDepth] = indexStack[levelDepth - 1] - 1;

            // add ID and calculate signature with the new index of the current level

            denseUnitIds[levelDepth] = coreSetIds[indexStack[levelDepth]];

            signatureStack[levelDepth + 1] =
                signatureStack[levelDepth] + labels[denseUnitIds[levelDepth]];
        }

        // Add current Dense Unit
        signature = signatureStack[minPoints];

        if (signature >= minSigBoundary && signature < maxSigBoundary)
        {
            denseUnitMap->insertDenseUnit(signature, denseUnitIds, dimension);
        }
    }
}

DenseUnitCreatorSeq::DenseUnitCreatorSeq(DenseUnitTable* denseUnitTable, int tableSize)
{
    this->tableSize = tableSize;
    this->denseUnitTable = denseUnitTable;
}

DenseUnitCreatorSeq::~DenseUnitCreatorSeq()
{
    free();
}

// calculate all dense units in given dimension
void DenseUnitCreatorSeq::calculate(unsigned long long minSigBoundary, unsigned long long maxSigBoundary, int dimension)
{
    for (int i = 0; i < coreSets[dimension].size(); i++)
    {
        denseUnitCalc(
            coreSets[dimension][i].ids,
            coreSets[dimension][i].size,
            coreSets[dimension][i].pivot,
            dimension,
            minSigBoundary,
            maxSigBoundary
        );
    }
}

// calculate all dense units all dimensions
void DenseUnitCreatorSeq::createDenseUnits(unsigned long long minSigBoundary, unsigned long long maxSigBoundary)
{
    for (int dim = 0; dim < coreSets.size(); dim++)
    {
        calculate(minSigBoundary, maxSigBoundary, dim);
    }
}

// initialze dense unit creator
void DenseUnitCreatorSeq::init(std::vector<std::vector<CoreSet>> coreSets, unsigned long long* labels, int numPoints, int minPoints)
{
	this->coreSets = coreSets;
	this->labels = labels;
    this->minPoints = minPoints;

    denseUnitIds = new unsigned int[minPoints];
    indexStack = new int[minPoints];
    signatureStack = new unsigned long long[minPoints + 1];

    denseUnitMap = new DenseUnitMapSeq(denseUnitTable, tableSize);

    clear();
}

// reset to initial state
void DenseUnitCreatorSeq::clear()
{
    denseUnitMap->clear();
}

// free memory
void DenseUnitCreatorSeq::free()
{
    delete[] denseUnitIds;
    delete[] indexStack;
    delete[] signatureStack;

    delete denseUnitMap;
}
