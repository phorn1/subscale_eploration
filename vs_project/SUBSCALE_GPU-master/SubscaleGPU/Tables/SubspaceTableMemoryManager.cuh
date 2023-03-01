#pragma once
#include "SubspaceTable.cuh"
#include "../MemoryManager/MemoryManager.cuh"

class SubspaceTableMemoryManager : public SubspaceTable, public virtual MemoryManager<SubspaceTable>
{
protected:

	SubspaceTable* allocMembers();

	void freeMembers();

public:
	SubspaceTableMemoryManager(int idsSize, int dimensionsSize, int tableSize) : SubspaceTable(idsSize, dimensionsSize, tableSize) {}

	void resetMembers();
};