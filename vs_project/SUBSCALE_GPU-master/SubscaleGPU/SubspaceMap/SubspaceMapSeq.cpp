#include "SubspaceMapSeq.h"

// Constructor
SubspaceMapSeq::SubspaceMapSeq(SubspaceTable* table, uint32_t tableSize)
{
	this->table = table;
	doubleHash = new DoubleHash<uint32_t>(tableSize);
}

// Inserts an entry into the table
void SubspaceMapSeq::insertEntry(uint32_t* ids, uint32_t* dimensions)
{
	std::pair<bool, int> got = doubleHash->findArray(
		dimensions, 
		table->getDimensions(), 
		table->getDimensionsSize());

	if (got.first)
	{
		// Entry already exists
		table->mergeIds(ids, got.second);
	}
	else
	{
		// Insert new entry
		table->insertDimensions(dimensions, got.second);
		table->mergeIds(ids, got.second);
	}
}

// Resets class to initial state
void SubspaceMapSeq::clear()
{
	// nothing to clear, because table is cleared outside of this class
}

// Frees memory
void SubspaceMapSeq::free()
{
	delete doubleHash;
}

