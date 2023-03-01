#include "SubspaceTableMemoryManager.cuh"

SubspaceTable* SubspaceTableMemoryManager::allocMembers()
{
	// Allocate ids
	allocArray(&ids, (unsigned long long) idsSize * tableSize);

	// Allocate dimensions
	allocArray(&dimensions, (unsigned long long) dimensionsSize * tableSize);


	return (SubspaceTable*)this;
}

void SubspaceTableMemoryManager::freeMembers()
{
	// Free ids
	freeArray(ids);

	// Free dimensions
	freeArray(dimensions);
}


void SubspaceTableMemoryManager::resetMembers()
{
	// Set ids to zero
	setArrayContent(ids, 0x00, (unsigned long long) idsSize * tableSize);

	// Set dimensions to zero
	setArrayContent(dimensions, 0x00, (unsigned long long) dimensionsSize * tableSize);
}