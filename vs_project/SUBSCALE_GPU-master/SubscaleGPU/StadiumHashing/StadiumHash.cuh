#pragma once
#include "cuda_runtime.h"
#include "../SubscaleTypes.h"
#include <assert.h>
#include <stdio.h>

#include "../HelperFunctions/roundingFunctions.h"
#include "../HelperFunctions/cudaHelperFunctions.cuh"

typedef unsigned int uint;

#define KEY_NOT_INSIDE_TABLE 2

// Implementation of Stadium Hashing without the access bit
template<typename keyT, typename valueT>
class StadiumHash {
protected:
	uint ticketSize;
	uint numPosBits;
	uint numInfoBits;

	uint* ticketBoard;
	uint tableSize;

	// Hash Functions
	template <uint nRounds, uint rShift, uint mulOp>
	__device__ uint hash(const keyT key);

	__device__ uint initHash(keyT key);
	__device__ uint rehash(uint hashed, const keyT key);
	__device__ uint infoHash(keyT key);

	// Ticket Board
	__host__ void allocTicketBoard(); // Allokieren und Memset
	__host__ void clearTicketBoard();
	__host__ void freeTicketBoard();
	__device__ uint prepareTicket(uint info, uint infoStart);
	__device__ uint extractInfo(uint ticket, uint infoStart);
	__device__ uint getTbIndex(uint hashed);
	__device__ uint getPosInInt(uint hashed);
	__device__ uint tryBookASeat(uint tbIndex, uint posInInt);
	__device__ uint tryFindTheSeat(keyT key, uint hashed, uint tbIndex, uint posInInt, uint info, keyT* keys);
	__device__ void insertTicketInfo(uint info, uint tbIndex, uint posInInt);


public:
	StadiumHash(uint tableSize, uint ticketSizeX);
	~StadiumHash();
	__device__ uint insert(keyT key, valueT value, keyT* keys, valueT* values);
	__device__ uint find(keyT key, keyT* keys);
};

