#pragma once
#include "../HelperFunctions/cudaHelperFunctions.cuh"
#include "TableConversionKernels.cuh"
#include <stdexcept>
#include "ITableManager.h"


class TableManager : public ITableManager {
private:
	int numberOfThreads;
public:
	TableManager(int numberOfThreads)
	{
		this->numberOfThreads = numberOfThreads;
	}

	void deviceToLocal(SubscaleTable* localTable, SubscaleTable* deviceTable, unsigned int numberOfEntries);

	void localToDevice(SubscaleTable* deviceTable, SubscaleTable* localTable, unsigned int numberOfEntries);

	void localToLocal(SubscaleTable* targetTable, SubscaleTable* sourceTable, unsigned int numberOfEntries);



	void filterIds(SubspaceTable* table, int tableSize, int minNumberOfIds);
	void filterDimensions(DenseUnitTable* table, int tableSize, int minNumberOfDimensions);
	unsigned int condenseTable(SubscaleTable* targetTable, SubscaleTable* sourceTable, int sourceTableSize);
	unsigned int copyDenseUnitToSubspaceTable(SubspaceTable* targetTable, DenseUnitTable* sourceTable, int sourceTableSize);

	unsigned int duToSSTable(SubspaceTable* targetTable, DenseUnitTable* sourceTable, int sourceTableSize);
	unsigned int ssToSSTable(SubspaceTable* targetTable, SubspaceTable* sourceTable, int sourceTableSize, int minNumberOfIds);

	unsigned int countEntries(SubscaleTable* table, int tableSize);

};


