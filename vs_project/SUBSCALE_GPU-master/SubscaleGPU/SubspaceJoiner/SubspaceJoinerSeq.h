#pragma once
#include "../SubspaceMap/SubspaceMapSeq.h"
#include "ISubspaceJoiner.h"

// Sequential implementation of a subspace joiner
class SubspaceJoinerSeq : public ISubspaceJoiner
{
private:

	// hash table for candidates
	SubspaceMapSeq* subspaceMap;
	

public:
	SubspaceJoinerSeq(SubspaceTable* sourceTable, int tableSize);
	void init(unsigned int ticketSize);
	void free();
	void clear();
	void join(SubspaceTable* subspaceTable, int ssTableSize);

};

