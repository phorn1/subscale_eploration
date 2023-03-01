#pragma once
#include "SubscaleTable.cuh"
#include "SubspaceTable.cuh"
#include "DenseUnitTable.cuh"

class ITableManager
{
public:
	virtual void deviceToLocal(SubscaleTable* localTable, SubscaleTable* deviceTable, unsigned int numberOfEntries) = 0;

	virtual void localToDevice(SubscaleTable* deviceTable, SubscaleTable* localTable, unsigned int numberOfEntries) = 0;

	virtual void localToLocal(SubscaleTable* targetTable, SubscaleTable* sourceTable, unsigned int numberOfEntries) = 0;

	virtual void filterIds(SubspaceTable* table, int tableSize, int minNumberOfIds) = 0;
	virtual void filterDimensions(DenseUnitTable* table, int tableSize, int minNumberOfDimensions) = 0;
	virtual unsigned int condenseTable(SubscaleTable* targetTable, SubscaleTable* sourceTable, int sourceTableSize) = 0;
	virtual unsigned int copyDenseUnitToSubspaceTable(SubspaceTable* targetTable, DenseUnitTable* sourceTable, int sourceTableSize) = 0;

	virtual unsigned int duToSSTable(SubspaceTable* targetTable, DenseUnitTable* sourceTable, int sourceTableSize) = 0;

	virtual unsigned int countEntries(SubscaleTable* table, int tableSize) = 0;
};
