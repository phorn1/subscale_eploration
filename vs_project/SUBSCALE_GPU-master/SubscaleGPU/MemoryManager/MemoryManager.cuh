#pragma once
#include "../HelperFunctions/cudaHelperFunctions.cuh"

// interface for memory management
template<typename ClassType>
class MemoryManager
{
protected:
	ClassType* ptr;

	virtual void alloc() = 0;
	virtual void free() = 0;

	virtual ClassType* allocMembers() = 0;
	virtual void freeMembers() = 0;
	virtual void resetMembers() = 0;

	virtual void allocArray(uint32_t** arr, uint64_t arrSize) = 0;
	virtual void allocArray(uint64_t** arr, uint64_t arrSize) = 0;

	virtual void copyArrayContent(uint32_t* targetArr, uint32_t* sourceArr, uint64_t arrSize) = 0;
	virtual void copyArrayContent(uint64_t* targetArr, uint64_t* sourceArr, uint64_t arrSize) = 0;
	virtual void setArrayContent(uint32_t* arr, uint8_t value, uint64_t arrSize) = 0;
	virtual void setArrayContent(uint64_t* arr, uint8_t value, uint64_t arrSize) = 0;
	virtual void freeArray(uint32_t* arr) = 0;
	virtual void freeArray(uint64_t* arr) = 0;

public:
	ClassType* getPtr();

	virtual void reset() = 0;

	virtual ~MemoryManager() {}

};