#pragma once
#include "DenseUnitTable.cuh"
#include "../MemoryManager/MemoryManager.cuh"

class DenseUnitTableMemoryManager : public DenseUnitTable, public virtual MemoryManager<DenseUnitTable>
{
protected:

	DenseUnitTable* allocMembers();

	void freeMembers();

public:
	DenseUnitTableMemoryManager(int idsSize, int dimensionsSize, int tableSize) : DenseUnitTable(idsSize, dimensionsSize, tableSize) {}

	void resetMembers();
};