#pragma once
#include "../Tables/SubspaceTable.cuh"

// Interface for subspace joiners
class ISubspaceJoiner 
{
protected:
	SubspaceTable* subspaceTable;
	int tableSize;

public:
	virtual void join(SubspaceTable* sourceTable, int ssTableSize) = 0;

	virtual void init(unsigned int ticketSize) = 0;

	virtual void free() = 0;
	virtual void clear() = 0;
};
