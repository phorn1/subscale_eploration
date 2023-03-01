#include "UnifiedMemory.cuh"
#include "../Tables/DenseUnitTable.cuh"
#include "../Tables/SubspaceTable.cuh"
#include "../DenseUnitMap/DenseUnitMap.cuh"
#include "../SubspaceMap/SubspaceMap.cuh"
#include "../BinomialCoeffCreator/BinomialCoeffCreator.cuh"




// allocates memory for members of a nested data structure and the data structure itself
template<typename ClassType>
void UnifiedMemory<ClassType>::alloc()
{
	cudaError_t cudaStatus;

	// Allocate member variables
	ClassType* hostPtr = this->allocMembers();

	// Allocate device pointer
	cudaStatus = cudaMallocManaged((void**)&(this->ptr), sizeof(ClassType));
	checkStatus(cudaStatus);

	// Copy member variables to device
	cudaStatus = cudaMemcpy(this->ptr, hostPtr, sizeof(ClassType), cudaMemcpyHostToDevice);
	checkStatus(cudaStatus);
}

// method to allocate an array
template<typename ClassType>
void UnifiedMemory<ClassType>::allocArray(uint32_t** arr, uint64_t arrSize)
{
	cudaError_t cudaStatus = cudaMallocManaged((void**)arr, arrSize * sizeof(uint32_t));
	checkStatus(cudaStatus);
}

// method to allocate an array
template<typename ClassType>
void UnifiedMemory<ClassType>::allocArray(uint64_t** arr, uint64_t arrSize)
{
	cudaError_t cudaStatus = cudaMallocManaged((void**)arr, arrSize * sizeof(uint64_t));
	checkStatus(cudaStatus);
}

// method to copy the content of an array
template<typename ClassType>
void UnifiedMemory<ClassType>::copyArrayContent(uint32_t* targetArr, uint32_t* sourceArr, uint64_t arrSize)
{
	copyArrayLocalToDevice(targetArr, sourceArr, arrSize);
}

// method to copy the content of an array
template<typename ClassType>
void UnifiedMemory<ClassType>::copyArrayContent(uint64_t* targetArr, uint64_t* sourceArr, uint64_t arrSize)
{
	copyArrayLocalToDevice(targetArr, sourceArr, arrSize);
}

// method to set the content of an array
template<typename ClassType>
void UnifiedMemory<ClassType>::setArrayContent(uint32_t* arr, uint8_t value, uint64_t arrSize)
{
	cudaError_t cudaStatus = cudaMemset((void*)arr, value, arrSize * sizeof(uint32_t));
	checkStatus(cudaStatus);
}

// method to set the content of an array
template<typename ClassType>
void UnifiedMemory<ClassType>::setArrayContent(uint64_t* arr, uint8_t value, uint64_t arrSize)
{
	cudaError_t cudaStatus = cudaMemset((void*)arr, value, arrSize * sizeof(uint64_t));
	checkStatus(cudaStatus);
}

// method to free the memory of an array
template<typename ClassType>
void UnifiedMemory<ClassType>::freeArray(uint32_t* arr)
{
	cudaError_t cudaStatus = cudaFree((void*)arr);
	checkStatus(cudaStatus);
}

// method to free the memory of an array
template<typename ClassType>
void UnifiedMemory<ClassType>::freeArray(uint64_t* arr)
{
	cudaError_t cudaStatus = cudaFree((void*)arr);
	checkStatus(cudaStatus);
}

// frees memory of members of a nested data structure and of the data structure itself
template<typename ClassType>
void UnifiedMemory<ClassType>::free()
{
	cudaError_t cudaStatus;

	this->freeMembers();

	cudaStatus = cudaFree((void*)(this->ptr));
	checkStatus(cudaStatus);
}

// resets members to the initial state
template<typename ClassType>
void UnifiedMemory<ClassType>::reset()
{
	this->resetMembers();
}

template class UnifiedMemory<DenseUnitTable>;
template class UnifiedMemory<SubspaceTable>;
template class UnifiedMemory<BinomialCoeffCreator>;
template class UnifiedMemory<DenseUnitMap>;
template class UnifiedMemory<SubspaceMap>;