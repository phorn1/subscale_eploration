#pragma once
#include "cuda_runtime.h"
#include <assert.h>
#include <stdio.h>

typedef unsigned int uint;

#define KEY_NOT_INSIDE_TABLE 2

// Implementation of Stadium Hashing with the access bit to allow concurrent inserts and searches
template<typename keyT, typename valueT>
class ConcStadiumHash {
protected:
	volatile uint* ticketBoard;
	uint tableSize;

	// Hash Functions
	template <uint nRounds, uint rShift, uint mulOp>
	__device__ uint hash(const keyT key);

	__device__ uint initHash(const keyT key);
	__device__ uint rehash(uint hashed, const keyT key);
	__device__ uint infoHash(keyT key);

	__device__ uint arrayHash(keyT* arr, uint length);
	__device__ uint arrayRehash(uint hashed, keyT* arr, uint length);
	__device__ uint arrayInfoHash(keyT* arr, uint length);

	
	// Ticket Board
	__host__ void allocTicketBoard();
	__host__ void clearTicketBoard();
	__host__ void freeTicketBoard();
	__device__ uint prepareTicket(uint info, uint infoStart);
	__device__ uint extractInfo(uint ticket, uint infoStart);
	__device__ uint getTbIndex(uint hashed);
	__device__ uint getPosInInt(uint hashed);
	__device__ uint tryBookASeat(uint tbIndex, uint posInInt);
	__device__ uint tryFindTheSeat(keyT* key, uint hashed, uint tbIndex, uint posInInt, uint info, keyT* keys, uint keyLength);
	__device__ uint tryFindTheSeat(keyT key, uint hashed, uint tbIndex, uint posInInt, uint info, keyT* keys);
	__device__ void insertTicketInfo(uint info, uint tbIndex, uint posInInt);

	__device__ uint isAccessed(uint tbIndex, uint posInInt);
	__device__ void unlockAccess(uint tbIndex, uint posInInt);


public:
	ConcStadiumHash(uint tableSize);
	~ConcStadiumHash();
	__device__ void insert(keyT* key, valueT value, keyT* keys, valueT* values, uint keyLength);
	__device__ uint find(keyT* key, keyT* keys, uint keyLength);
};



