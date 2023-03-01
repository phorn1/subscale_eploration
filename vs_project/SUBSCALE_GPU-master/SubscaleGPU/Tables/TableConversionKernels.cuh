#pragma once
#include <device_launch_parameters.h>
#include "SubspaceTable.cuh"
#include "DenseUnitTable.cuh"

__global__ void tableDimensionsFilterKernel(DenseUnitTable* table, int minNumberOfDims);
__global__ void tableIdsFilterKernel(SubspaceTable* table, int minNumberOfIds);
__global__ void tableCondenseKernel(SubscaleTable* targetTable, SubscaleTable* sourceTable, unsigned int* index);
__global__ void tableCondenseKernel(SubspaceTable* targetTable, DenseUnitTable* sourceTable, unsigned int* index);
__global__ void tableCountKernel(SubscaleTable* table, unsigned int* count);

// __global__ void denseUnitToSubspaceKernel(SubspaceTable* targetTable, DenseUnitTable* sourceTable, unsigned int* index);
// __global__ void subspaceToSubspaceKernel(SubspaceTable* targetTable, SubspaceTable* sourceTable, int minIds, unsigned int* index);


