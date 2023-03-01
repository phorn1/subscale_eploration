#pragma once
#include "MemoryManager.cuh"
#include <cstring>

// adapter class for local host memory
template<typename ClassType>
class LocalMemory : public virtual MemoryManager<ClassType> {

protected:
	void alloc();

	void allocArray(uint32_t** arr, uint64_t arrSize);
	void allocArray(uint64_t** arr, uint64_t arrSize);

	void copyArrayContent(uint32_t* targetArr, uint32_t* sourceArr, uint64_t arrSize);
	void copyArrayContent(uint64_t* targetArr, uint64_t* sourceArr, uint64_t arrSize);
	void setArrayContent(uint32_t* arr, uint8_t value, uint64_t arrSize);
	void setArrayContent(uint64_t* arr, uint8_t value, uint64_t arrSize);

	void freeArray(uint32_t* arr);
	void freeArray(uint64_t* arr);

	void free();

public:

	void reset();

};
