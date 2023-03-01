#include "HostMemory.cuh"
#include "../Tables/DenseUnitTable.cuh"
#include "../Tables/SubspaceTable.cuh"
#include "../DenseUnitMap/DenseUnitMap.cuh"
#include "../SubspaceMap/SubspaceMap.cuh"
#include "../BinomialCoeffCreator/BinomialCoeffCreator.cuh"



// allocates memory for members of a nested data structure and the data structure itself
template<typename ClassType>
void HostMemory<ClassType>::alloc()
{
	cudaError_t cudaStatus;

	// Allocate member variables
	ClassType* hostPtr = this->allocMembers();

	// Allocate device pointer
	cudaStatus = cudaHostAlloc((void**)&(this->ptr), sizeof(ClassType), cudaHostRegisterMapped);
	checkStatus(cudaStatus);

	// Copy member variables to device
	cudaStatus = cudaMemcpy(this->ptr, hostPtr, sizeof(ClassType), cudaMemcpyHostToDevice);
	checkStatus(cudaStatus);
}

// method to allocate an array
template<typename ClassType>
void HostMemory<ClassType>::allocArray(uint32_t** arr, uint64_t arrSize)
{
	cudaError_t cudaStatus = cudaHostAlloc((void**)arr, arrSize * sizeof(uint32_t), cudaHostRegisterMapped);
	checkStatus(cudaStatus);
}

// method to allocate an array
template<typename ClassType>
void HostMemory<ClassType>::allocArray(uint64_t** arr, uint64_t arrSize)
{
	cudaError_t cudaStatus = cudaHostAlloc((void**)arr, arrSize * sizeof(uint64_t), cudaHostRegisterMapped);
	checkStatus(cudaStatus);
}

// method to copy the content of an array
template<typename ClassType>
void HostMemory<ClassType>::copyArrayContent(uint32_t* targetArr, uint32_t* sourceArr, uint64_t arrSize)
{
	copyArrayLocalToDevice(targetArr, sourceArr, arrSize);
}

// method to copy the content of an array
template<typename ClassType>
void HostMemory<ClassType>::copyArrayContent(uint64_t* targetArr, uint64_t* sourceArr, uint64_t arrSize)
{
	copyArrayLocalToDevice(targetArr, sourceArr, arrSize);
}

// method to set the content of an array
template<typename ClassType>
void HostMemory<ClassType>::setArrayContent(uint32_t* arr, uint8_t value, uint64_t arrSize)
{
	cudaError_t cudaStatus = cudaMemset((void*)arr, value, arrSize * sizeof(uint32_t));
	checkStatus(cudaStatus);
}

// method to set the content of an array
template<typename ClassType>
void HostMemory<ClassType>::setArrayContent(uint64_t* arr, uint8_t value, uint64_t arrSize)
{
	cudaError_t cudaStatus = cudaMemset((void*)arr, value, arrSize * sizeof(uint64_t));
	checkStatus(cudaStatus);
}

// method to free the memory of an array
template<typename ClassType>
void HostMemory<ClassType>::freeArray(uint32_t* arr)
{
	cudaError_t cudaStatus = cudaFreeHost(arr);
	checkStatus(cudaStatus);
}

// method to free the memory of an array
template<typename ClassType>
void HostMemory<ClassType>::freeArray(uint64_t* arr)
{
	cudaError_t cudaStatus = cudaFreeHost(arr);
	checkStatus(cudaStatus);
}

// frees memory of members of a nested data structure and of the data structure itself
template<typename ClassType>
void HostMemory<ClassType>::free()
{
	this->freeMembers();

	cudaError_t cudaStatus = cudaFreeHost(this->ptr);
	checkStatus(cudaStatus);
}

// resets members to the initial state
template<typename ClassType>
void HostMemory<ClassType>::reset()
{
	this->resetMembers();
}

template class HostMemory<DenseUnitTable>;
template class HostMemory<SubspaceTable>;
template class HostMemory<BinomialCoeffCreator>;
template class HostMemory<DenseUnitMap>;
template class HostMemory<SubspaceMap>;