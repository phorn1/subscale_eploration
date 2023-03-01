#include "SubspaceJoinerSeq.h"

// Constructor
SubspaceJoinerSeq::SubspaceJoinerSeq(SubspaceTable* sourceTable, int tableSize)
{
	subspaceTable = sourceTable;
	this->tableSize = tableSize;
}

// Initializes hash map
void SubspaceJoinerSeq::init(unsigned int ticketSize)
{
	subspaceMap = new SubspaceMapSeq(subspaceTable, tableSize);
}

// Deletes hash map
void SubspaceJoinerSeq::free()
{
	delete subspaceMap;
}

// Clears hash map
void SubspaceJoinerSeq::clear()
{
	subspaceMap->clear();
}

// Inserts all entries from a table into the hash map and thereby joins them to cluster candidates
void SubspaceJoinerSeq::join(SubspaceTable* subspaceTable, int ssTableSize)
{
	for (int i = 0; i < ssTableSize; i++)
	{
		unsigned int* ids = subspaceTable->getIds(i);
		unsigned int* dimensions = subspaceTable->getDimensions(i);

		subspaceMap->insertEntry(ids, dimensions);
	}
}
