#include "../DenseUnitMap/DenseUnitMap.cuh"
#include "../SubspaceMap/SubspaceMap.cuh"
#include "../SubspaceMap/DeviceSubspaceMap.cuh"
#include "ISubspaceJoiner.h"
#include "../MemoryManager/MemoryManager.cuh"

// Parallel implementation of a subspace joiner
class SubspaceJoiner : public ISubspaceJoiner
{
private:
	int numberOfThreads;

	// hash table for candidates
	MemoryManager<SubspaceMap>* subspaceMapWrapper;

public:
	
	SubspaceJoiner(SubspaceTable* subspaceTable, int tableSize, int numberOfThreads);
	void init(unsigned int ticketSize);
	void free();
	void clear();
	void join(SubspaceTable* sourceTable, int ssTableSize);	
};


