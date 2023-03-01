#pragma once
#include "ITableManager.h"

class TableManagerSeq : public ITableManager
{
private:
	int countBits(unsigned int value);

public:
	void deviceToLocal(SubscaleTable* localTable, SubscaleTable* deviceTable, unsigned int numberOfEntries);

	void localToDevice(SubscaleTable* deviceTable, SubscaleTable* localTable, unsigned int numberOfEntries);

	void localToLocal(SubscaleTable* targetTable, SubscaleTable* sourceTable, unsigned int numberOfEntries);

	void filterIds(SubspaceTable* table, int tableSize, int minNumberOfIds);
	void filterDimensions(DenseUnitTable* table, int tableSize, int minNumberOfDimensions);
	unsigned int condenseTable(SubscaleTable* targetTable, SubscaleTable* sourceTable, int sourceTableSize);
	unsigned int copyDenseUnitToSubspaceTable(SubspaceTable* targetTable, DenseUnitTable* sourceTable, int sourceTableSize);

	unsigned int duToSSTable(SubspaceTable* targetTable, DenseUnitTable* sourceTable, int sourceTableSize);

	unsigned int countEntries(SubscaleTable* table, int tableSize);
};

