#include "BitMapper.cuh"

// Get the index of a field in the bit array. The field contains the bit corresponding to value.
__host__ __device__ unsigned int BitMapper::getIndexInBitArray(unsigned int value)
{
	// 5 = log2(32)
	return (value >> 5); // 5 Bits, to fit 32 different positions
}

// Get the position of a bit in a field in the bit array. The bit corresponds to value.
__host__ __device__ unsigned int BitMapper::getPosInBitField(unsigned int value)
{
	// 31 = 32-1
	return (value & 31); // 31 = 11111 -> five 1s
}

// Converts a bit array to a vector
vector<unsigned int> BitMapper::bitArrayToVector(unsigned int* bitArray, unsigned int arraySize)
{

	vector<unsigned int> vec;
	unsigned int currentField;

	for (unsigned int i = 0; i < arraySize; i++)
	{
		currentField = bitArray[i];
		unsigned int j = 0;

		while (currentField)
		{
			if (currentField & 0x1)
			{
				vec.push_back(i * 32 + j);
			}

			j++;
			currentField = currentField >> 1;
		}
	}

	return vec;
}

// Adds a value to a bit array
__host__ void BitMapper::addValueToBitArray(unsigned int value, unsigned int* bitArray)
{
	unsigned int arrayIndex = getIndexInBitArray(value);
	int posInField = getPosInBitField(value);

	bitArray[arrayIndex] |= (1 << posInField);
}

// Merges two bit arrays so that all set bits are contained in one array
__host__ void BitMapper::mergeBitArrays(unsigned int* targetArray, unsigned int* sourceArray, unsigned int arraySize)
{
	for (int i = 0; i < arraySize; i++)
	{
		targetArray[i] |= sourceArray[i];
	}
}

// Adds a value to a bit array on the device (requires atomic operation)
__device__ void BitMapper::dev_addValueToBitArray(unsigned int value, unsigned int* bitArray)
{
	unsigned int arrayIndex = getIndexInBitArray(value);
	int posInField = getPosInBitField(value);

	atomicOr(bitArray + arrayIndex, (1 << posInField));
}

// Merges two bit arrays on the device (requires atomic operation)
__device__ void BitMapper::dev_mergeBitArrays(unsigned int* targetArray, unsigned int* sourceArray, unsigned int arraySize)
{
	for (int i = 0; i < arraySize; i++)
	{
		atomicOr(targetArray + i, sourceArray[i]);
	}
}

