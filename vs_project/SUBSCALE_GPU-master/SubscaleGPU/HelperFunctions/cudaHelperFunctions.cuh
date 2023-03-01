#pragma once
#include <stdexcept>
#include "cuda_runtime.h"

// helper functions for accessing the CUDA runtime api

template<typename arrayType>
void copyArrayDeviceToLocal(arrayType* target_arr, arrayType* source_arr, uint64_t arrSize);

template<typename arrayType>
void copyArrayLocalToDevice(arrayType* target_arr, arrayType* source_arr, uint64_t arrSize);

template<typename arrayType>
void copyArrayDeviceToDevice(arrayType* target_arr, arrayType* source_arr, uint64_t arrSize);

void synchronizeKernelCall();

void checkStatus(cudaError_t cudaStatus);
