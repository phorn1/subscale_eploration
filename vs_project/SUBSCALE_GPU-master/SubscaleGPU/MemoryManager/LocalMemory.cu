#include "LocalMemory.cuh"
#include "../Tables/DenseUnitTable.cuh"
#include "../Tables/SubspaceTable.cuh"
#include "../DenseUnitMap/DenseUnitMap.cuh"
#include "../SubspaceMap/SubspaceMap.cuh"
#include "../BinomialCoeffCreator/BinomialCoeffCreator.cuh"


// allocates memory for members of a nested data structure
template<typename ClassType>
void LocalMemory<ClassType>::alloc() {
	this->ptr = this->allocMembers();
}

// method to allocate an array
template<typename ClassType>
void LocalMemory<ClassType>::allocArray(uint32_t** arr, uint64_t arrSize) {

	*arr = new uint32_t[arrSize];
}

// method to allocate an array
template<typename ClassType>
void LocalMemory<ClassType>::allocArray(uint64_t** arr, uint64_t arrSize)
{
	*arr = new uint64_t[arrSize];
}

// method to copy the content of an array
template<typename ClassType>
void LocalMemory<ClassType>::copyArrayContent(uint32_t* targetArr, uint32_t* sourceArr, uint64_t arrSize)
{
	memcpy(targetArr, sourceArr, arrSize * sizeof(uint32_t));
}

// method to copy the content of an array
template<typename ClassType>
void LocalMemory<ClassType>::copyArrayContent(uint64_t* targetArr, uint64_t* sourceArr, uint64_t arrSize)
{
	memcpy(targetArr, sourceArr, arrSize * sizeof(uint64_t));
}

// method to set the content of an array
template<typename ClassType>
void LocalMemory<ClassType>::setArrayContent(uint32_t* arr, uint8_t value, uint64_t arrSize)
{
	memset(arr, value, arrSize * sizeof(uint32_t));
}

// method to set the content of an array
template<typename ClassType>
void LocalMemory<ClassType>::setArrayContent(uint64_t* arr, uint8_t value, uint64_t arrSize)
{
	memset(arr, value, arrSize * sizeof(uint64_t));
}

// method to free the memory of an array
template<typename ClassType>
void LocalMemory<ClassType>::freeArray(uint32_t* arr)
{
	delete[] arr;
}

// method to free the memory of an array
template<typename ClassType>
void LocalMemory<ClassType>::freeArray(uint64_t* arr)
{
	delete[] arr;
}

// frees memory of members of a nested data structure
template<typename ClassType>
void LocalMemory<ClassType>::free() {
	this->freeMembers();
}

// resets members to the initial state
template<typename ClassType>
void LocalMemory<ClassType>::reset()
{
	this->resetMembers();
}

template class LocalMemory<DenseUnitTable>;
template class LocalMemory<SubspaceTable>;
template class LocalMemory<BinomialCoeffCreator>;
template class LocalMemory<DenseUnitMap>;
template class LocalMemory<SubspaceMap>;