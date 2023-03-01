#pragma once
#include "device_launch_parameters.h"
#include "../DenseUnitMap/DenseUnitMap.cuh"
#include "../BinomialCoeffCreator/BinomialCoeffCreator.cuh"


// kernels for dense unit creation

// kernel for creating a single dense unit per thread (DEPRECATED)
__global__ void denseUnitKernelSingle(
    const unsigned int* coreSetIds,
    const int coreSetSize,
    const int dimension,
    const int minPoints,
    const unsigned long long startIndex,
    const unsigned long long endIndex,
    const unsigned long long minSigBoundary,
    const unsigned long long maxSigBoundary,
    const unsigned long long* labels,
    DenseUnitMap* denseUnitMap,
    BinomialCoeffCreator* binomCoeffs);

// kernel for creating multiple dense units per thread
__global__ void denseUnitKernelMulti(
    const unsigned int* coreSetIds,
    const int coreSetSize,
    const int dimension,
    const int minPoints,
    const unsigned long long startIndex,
    const unsigned long long endIndex,
    const unsigned long long minSigBoundary,
    const unsigned long long maxSigBoundary,
    const unsigned long long* labels,
    DenseUnitMap* denseUnitMap,
    BinomialCoeffCreator* binomCoeffs,
    int denseUnitsPerThread);