#include "DenseUnitMapMemoryManager.cuh"

DenseUnitMap* DenseUnitMapMemoryManager::allocMembers()
{
	// Allocate keys
	allocArray(&keys, tableSize);

	// Allocate ticket board
	allocTicketBoard();

	return (DenseUnitMap*) this;
}

void DenseUnitMapMemoryManager::freeMembers()
{
	// Free keys
	freeArray(keys);

	// Free ticket board
	freeTicketBoard();
}

void DenseUnitMapMemoryManager::resetMembers()
{
	// set keys to zero
	setArrayContent(keys, 0x00, tableSize);

	// reset ticket board
	clearTicketBoard();
}