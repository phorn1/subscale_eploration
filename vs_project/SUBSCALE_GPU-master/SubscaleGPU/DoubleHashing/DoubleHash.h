#pragma once

#include <utility>

#define KEY_NOT_INSIDE_TABLE 2

typedef unsigned int uint;

// sequential double hashing
template<typename keyT>
class DoubleHash
{
private:
	int tableSize;

	template <uint nRounds, uint rShift, uint mulOp>
	uint hash(const keyT key);

	uint initHash(keyT key);
	uint rehash(uint hashed, const keyT key);
	uint arrayHash(keyT* arr, uint length);
	uint arrayRehash(uint hashed, keyT* arr, uint length);

	bool isPrime(int number);

public:
	DoubleHash(int tableSize);
	std::pair<bool, int> find(keyT key, keyT* keys);
	std::pair<bool, int> findArray(keyT* key, keyT* keys, uint keyLength);
};
