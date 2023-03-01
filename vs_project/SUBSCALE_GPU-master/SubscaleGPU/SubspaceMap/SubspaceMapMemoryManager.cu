#include "SubspaceMapMemoryManager.cuh"

SubspaceMap* SubspaceMapMemoryManager::allocMembers()
{
	// Allocate ticket board
	allocTicketBoard();

	return (SubspaceMap*) this;
}

void SubspaceMapMemoryManager::freeMembers()
{
	// Free ticket board
	freeTicketBoard();
}

void SubspaceMapMemoryManager::resetMembers()
{
	// reset ticket board
	clearTicketBoard();
}