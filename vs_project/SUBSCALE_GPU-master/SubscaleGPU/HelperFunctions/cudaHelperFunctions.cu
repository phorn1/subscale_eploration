#include "cudaHelperFunctions.cuh"

// copies array form device to local host memory
template<typename arrayType>
void copyArrayDeviceToLocal(arrayType* target_arr, arrayType* source_arr, uint64_t arrSize)
{
	cudaError_t cudaStatus = cudaMemcpy(target_arr, source_arr, arrSize * sizeof(arrayType), cudaMemcpyDeviceToHost);
	checkStatus(cudaStatus);
}

// copies array form local host to device memory
template<typename arrayType>
void copyArrayLocalToDevice(arrayType* target_arr, arrayType* source_arr, uint64_t arrSize)
{
	cudaError_t cudaStatus = cudaMemcpy(target_arr, source_arr, arrSize * sizeof(arrayType), cudaMemcpyHostToDevice);
	checkStatus(cudaStatus);
}

// copies array form device to device memory
template<typename arrayType>
void copyArrayDeviceToDevice(arrayType* target_arr, arrayType* source_arr, uint64_t arrSize)
{
	cudaError_t cudaStatus = cudaMemcpy(target_arr, source_arr, arrSize * sizeof(arrayType), cudaMemcpyDeviceToDevice);
	checkStatus(cudaStatus);
}

// synchronizes all running streams
void synchronizeKernelCall()
{
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	checkStatus(cudaStatus);
}

// checks for an error in the execution of a CUDA method 
void checkStatus(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(cudaStatus));
	}
}


template void copyArrayDeviceToLocal<uint32_t>(uint32_t* target_arr, uint32_t* source_arr, uint64_t arrSize);
template void copyArrayDeviceToLocal<uint64_t>(uint64_t* target_arr, uint64_t* source_arr, uint64_t arrSize);
template void copyArrayLocalToDevice<uint32_t>(uint32_t* target_arr, uint32_t* source_arr, uint64_t arrSize);
template void copyArrayLocalToDevice<uint64_t>(uint64_t* target_arr, uint64_t* source_arr, uint64_t arrSize);
template void copyArrayDeviceToDevice<uint32_t>(uint32_t* target_arr, uint32_t* source_arr, uint64_t arrSize);
template void copyArrayDeviceToDevice<uint64_t>(uint64_t* target_arr, uint64_t* source_arr, uint64_t arrSize);