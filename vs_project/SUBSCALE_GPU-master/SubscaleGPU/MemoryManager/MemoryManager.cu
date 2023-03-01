#include "MemoryManager.cuh"

#include "../Tables/DenseUnitTable.cuh"
#include "../Tables/SubspaceTable.cuh"
#include "../DenseUnitMap/DenseUnitMap.cuh"
#include "../SubspaceMap/SubspaceMap.cuh"
#include "../BinomialCoeffCreator/BinomialCoeffCreator.cuh"


// get pointer of the object that is wrapped inside of memory manager
template<typename ClassType>
ClassType* MemoryManager<ClassType>::getPtr() {
	return ptr;
}

template class MemoryManager<DenseUnitTable>;
template class MemoryManager<SubspaceTable>;
template class MemoryManager<BinomialCoeffCreator>;
template class MemoryManager<DenseUnitMap>;
template class MemoryManager<SubspaceMap>;