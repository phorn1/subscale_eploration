#include "TableManager.cuh"
#include <cstring>


// Copys a table from device memory to local host memory
void TableManager::deviceToLocal(SubscaleTable* localTable, SubscaleTable* deviceTable, unsigned int numberOfEntries)
{
	cudaError_t cudaStatus;

	// Copy table object to host to get ids and dimensions pointers
	SubscaleTable* tmpLocalTable = new SubscaleTable(0, 0, 0);
	cudaStatus = cudaMemcpy(tmpLocalTable, deviceTable, sizeof(SubscaleTable), cudaMemcpyDeviceToHost);
	checkStatus(cudaStatus);

	int idsSize = tmpLocalTable->getIdsSize();
	int dimensionsSize = tmpLocalTable->getDimensionsSize();
	unsigned int* dev_ids = tmpLocalTable->getIds();
	unsigned int* dev_dimensions = tmpLocalTable->getDimensions();

	if (idsSize != localTable->getIdsSize() ||
		dimensionsSize != localTable->getDimensionsSize())
	{
		throw std::runtime_error("Table manager error: deviceToLocal() requires tables of the same size!");
	}

	// Copy ids
	copyArrayDeviceToLocal(localTable->getIds(), dev_ids, idsSize * numberOfEntries);
	
	// Copy dimensions
	copyArrayDeviceToLocal(localTable->getDimensions(), dev_dimensions, dimensionsSize * numberOfEntries);

	delete tmpLocalTable;
}

// Copys a table from local host memory to device memory
void TableManager::localToDevice(SubscaleTable* deviceTable, SubscaleTable* localTable, unsigned int numberOfEntries)
{
	cudaError_t cudaStatus;

	// Copy table object to host to get ids and dimensions pointers
	SubscaleTable* tmpLocalTable = new SubscaleTable(0, 0, 0);
	cudaStatus = cudaMemcpy(tmpLocalTable, deviceTable, sizeof(SubscaleTable), cudaMemcpyDeviceToHost);
	checkStatus(cudaStatus);

	int idsSize = tmpLocalTable->getIdsSize();
	int dimensionsSize = tmpLocalTable->getDimensionsSize();
	unsigned int* dev_ids = tmpLocalTable->getIds();
	unsigned int* dev_dimensions = tmpLocalTable->getDimensions();

	if (idsSize != localTable->getIdsSize() ||
		dimensionsSize != localTable->getDimensionsSize())
	{
		throw std::runtime_error("Table manager error: localToDevice() requires tables of the same size!");
	}

	// Copy ids
	copyArrayLocalToDevice(dev_ids, localTable->getIds(), idsSize * numberOfEntries);

	// Copy dimensions
	copyArrayLocalToDevice(dev_dimensions, localTable->getDimensions(), dimensionsSize * numberOfEntries);
}

// Copys a table from local host memory to local host memory
void TableManager::localToLocal(SubscaleTable* targetTable, SubscaleTable* sourceTable, unsigned int numberOfEntries)
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



// Filter all entries with fewer IDs than minNumberOfIds
void TableManager::filterIds(SubspaceTable* table, int tableSize, int minNumberOfIds)
{
	const int numberOfBlocks = tableSize / numberOfThreads + 1;

	tableIdsFilterKernel<<<numberOfBlocks, numberOfThreads>>>(table, minNumberOfIds);
	synchronizeKernelCall();
}

// Filter all entries with fewer dimensions than minNumberOfDimensions
void TableManager::filterDimensions(DenseUnitTable* table, int tableSize, int minNumberOfDimensions)
{
	const int numberOfBlocks = tableSize / numberOfThreads + 1;

	tableDimensionsFilterKernel << <numberOfBlocks, numberOfThreads >> > (table, minNumberOfDimensions);
	synchronizeKernelCall();
}

unsigned int TableManager::condenseTable(SubscaleTable* targetTable, SubscaleTable* sourceTable, int sourceTableSize)
{
	unsigned int* indexPtr;

	cudaError_t cudaStatus = cudaMalloc(&indexPtr, sizeof(unsigned int));
	checkStatus(cudaStatus);

	cudaStatus = cudaMemset(indexPtr, 0, sizeof(unsigned int));

	const int numberOfBlocks = sourceTableSize / numberOfThreads + 1;
	tableCondenseKernel << <numberOfBlocks, numberOfThreads >> > (targetTable, sourceTable, indexPtr);
	synchronizeKernelCall();

	unsigned int index;
	cudaStatus = cudaMemcpy(&index, indexPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	checkStatus(cudaStatus);

	return index;
}

unsigned int TableManager::copyDenseUnitToSubspaceTable(SubspaceTable* targetTable, DenseUnitTable* sourceTable, int sourceTableSize)
{
	const int numberOfBlocks = sourceTableSize / numberOfThreads + 1;
	unsigned int* indexPtr;

	cudaError_t cudaStatus = cudaMalloc(&indexPtr, sizeof(unsigned int));
	checkStatus(cudaStatus);

	cudaStatus = cudaMemset(indexPtr, 0, sizeof(unsigned int));


	tableCondenseKernel << <numberOfBlocks, numberOfThreads >> > (targetTable, sourceTable, indexPtr);

	synchronizeKernelCall();

	unsigned int index;
	cudaStatus = cudaMemcpy(&index, indexPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	checkStatus(cudaStatus);

	return index;
}


unsigned int TableManager::duToSSTable(SubspaceTable* targetTable, DenseUnitTable* sourceTable, int sourceTableSize)
{
	filterDimensions(sourceTable, sourceTableSize, 2);

	unsigned int index = copyDenseUnitToSubspaceTable(targetTable, sourceTable, sourceTableSize);

	
	return index;
}

unsigned int TableManager::ssToSSTable(SubspaceTable* targetTable, SubspaceTable* sourceTable, int sourceTableSize, int minNumberOfIds)
{
	filterIds(sourceTable, sourceTableSize, minNumberOfIds);
	unsigned int index = condenseTable(targetTable, sourceTable, sourceTableSize);

	return index;
}

unsigned int TableManager::countEntries(SubscaleTable* table, int tableSize)
{
	unsigned int* countPtr;

	cudaError_t cudaStatus = cudaMalloc(&countPtr, sizeof(unsigned int));
	checkStatus(cudaStatus);

	cudaStatus = cudaMemset(countPtr, 0, sizeof(unsigned int));

	const int numberOfBlocks = tableSize / numberOfThreads + 1;
	tableCountKernel << <numberOfBlocks, numberOfThreads >> > (table, countPtr);
	synchronizeKernelCall();

	unsigned int index;
	cudaStatus = cudaMemcpy(&index, countPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	checkStatus(cudaStatus);

	return index;
}
