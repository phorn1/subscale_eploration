#include "StadiumHash.cuh"

#define TICKET_SIZE 4
#define NUM_POS_BITS 3 // log(32/TICKET_SIZE)
#define NUM_INFO_BITS 3 // TICKET_SIZE - NUM_POS_BITS

// Generic Hash Function
template<typename keyT, typename valueT>
template <uint nRounds, uint rShift, uint mulOp>
__device__ uint StadiumHash<keyT, valueT>::hash(const keyT key)
{
	keyT x = key;
	for (uint j = nRounds; j > 0; --j) {
		x = ((x >> rShift) ^ x) * mulOp + j;
	}
	return (uint)x;
}

// First Hash 
template<typename keyT, typename valueT>
__device__ uint StadiumHash<keyT, valueT>::initHash(keyT key)
{
	uint hashed = hash<sizeof(keyT), 8, 0xE9D58A6B>(key);
	return __umulhi(hashed, tableSize);
}

// Second Hash
template<typename keyT, typename valueT>
__device__ uint StadiumHash<keyT, valueT>::rehash(uint hashed, const keyT key)
{
	uint h_2 = hash<sizeof(keyT), 8, 0x6E5B9D8A>(key);
	uint dh = hashed + 1 + __umulhi(h_2, tableSize - 1);
	return (dh >= tableSize) ? (dh - tableSize) : dh;
}

// Info Hash
template<typename keyT, typename valueT>
__device__ uint StadiumHash<keyT, valueT>::infoHash(keyT key)
{
	return hash<sizeof(keyT), 8, 0xCA34BE7D>(key) >> (32 - NUM_INFO_BITS); // mulOp was chosen randomly
}

// Allocation of the ticket board
template<typename keyT, typename valueT>
__host__ void StadiumHash<keyT, valueT>::allocTicketBoard()
{
	const int ticketBoardSize = tableSize / (32 / TICKET_SIZE) + 1;

	// Allocate memory on device for ticket board
	cudaError_t cudaStatus = cudaMalloc((void**)&ticketBoard, ticketBoardSize * sizeof(uint));
	checkStatus(cudaStatus);
}

// Clearing of the ticket board
template<typename keyT, typename valueT>
__host__ void StadiumHash<keyT, valueT>::clearTicketBoard()
{
	const int ticketBoardSize = tableSize / (32 / TICKET_SIZE) + 1;

	// clear ticket board by filling all tickets with 1s
	cudaError_t cudaStatus = cudaMemset((void*)ticketBoard, 0xFF, ticketBoardSize * sizeof(uint));
	checkStatus(cudaStatus);
}

// Deletion of the ticket board
template<typename keyT, typename valueT>
__host__ void StadiumHash<keyT, valueT>::freeTicketBoard()
{
	cudaFree((uint*) ticketBoard);
}

// Creates a mask containing info starting at infoStart and otherwise 1s
template<typename keyT, typename valueT>
__device__ uint StadiumHash<keyT, valueT>::prepareTicket(uint info, uint infoStart)
{
	uint mask = (1 << NUM_INFO_BITS) - 1;
	mask = ~(mask << infoStart);

	uint result = mask | (info << infoStart);

	return result;
}

// Extracts the info from a ticket
template<typename keyT, typename valueT>
__device__ uint StadiumHash<keyT, valueT>::extractInfo(uint ticket, uint infoStart)
{
	uint mask = (1 << NUM_INFO_BITS) - 1;
	mask = mask << infoStart;
	uint result = (mask & ticket) >> infoStart;

	return result;
}

// Calculates Ticket Board Index from the hashed key
template<typename keyT, typename valueT>
__device__ uint StadiumHash<keyT, valueT>::getTbIndex(uint hashed)
{
	return hashed >> NUM_POS_BITS;
}

// Calculates the Position of the ticket in the Ticket Board Entry from the hashed key
template<typename keyT, typename valueT>
__device__ uint StadiumHash<keyT, valueT>::getPosInInt(uint hashed)
{
	uint mask = (1 << NUM_POS_BITS) - 1;

	return (hashed & mask) << (5 - NUM_POS_BITS);
}

// Entry Reservation
template<typename keyT, typename valueT>
__device__ uint StadiumHash<keyT, valueT>::tryBookASeat(uint tbIndex, uint posInInt)
{
	uint permit = ((1 << (TICKET_SIZE - NUM_INFO_BITS)) - 1) << posInInt;
	uint auth = atomicAnd((uint*) (ticketBoard + tbIndex), ~permit);
	return (auth & permit) ? (~0) : 0;
}

// Entry Search
template<typename keyT, typename valueT>
__device__ uint StadiumHash<keyT, valueT>::tryFindTheSeat(keyT key, uint hashed, uint tbIndex, uint posInInt, uint info, keyT* keys)
{
	uint ticketHolder = ticketBoard[tbIndex];
	uint permit = 1 << posInInt;

	// check if bucket is empty
	if (permit & ticketHolder) {
		return KEY_NOT_INSIDE_TABLE;
	}
	else {
		// get and compare info from ticket
		uint retrievedInfo = extractInfo(ticketHolder, posInInt + (TICKET_SIZE - NUM_INFO_BITS));
		if (info != retrievedInfo) return 0;

		// compare keys
		return (keys[hashed] == key) ? (~0) : 0;
	}
}

// Inserts info into the ticket board
template<typename keyT, typename valueT>
__device__ void StadiumHash<keyT, valueT>::insertTicketInfo(uint info, uint tbIndex, uint posInInt)
{
	uint prepTicket = prepareTicket(info, posInInt + 1);
	atomicAnd((uint*) (ticketBoard + tbIndex), prepTicket);
}

// Constructor
template<typename keyT, typename valueT>
StadiumHash<keyT, valueT>::StadiumHash(uint tableSize, uint ticketSize)
{
	if ((sizeof(uint) * CHAR_BIT) % ticketSize != 0 || ticketSize == 1)
	{
		throw std::runtime_error("Ticket size for stadium hash has to be larger than 1 and devide 32 evenly!\n");
	}

	this->tableSize = roundToNextPrime(tableSize);

	/* NOTE
	This version of stadium hashing doesn't use a configurable ticket size. Tests have shown that accessing the
	ticket size as a variable is slower than using a fixed defined ticket size. If flexibility is more important
	than performance, these existing variables can be used:

	this->ticketSize = ticketSize;
	this->numInfoBits = ticketSize - 1;
	this->numPosBits = log2((sizeof(uint) * CHAR_BIT) / ticketSize);
	*/
}

// Destructor
template<typename keyT, typename valueT>
StadiumHash<keyT, valueT>::~StadiumHash()
{
}

// Inserts a given key value pair into the table
template<typename keyT, typename valueT>
__device__ uint StadiumHash<keyT, valueT>::insert(keyT key, valueT value, keyT* keys, valueT* values)
{
	// Initial Hash
	uint hashed = initHash(key);

	// Get ticket board index and the position in the integer from hashed value
	uint tbIndex = getTbIndex(hashed);
	uint posInInt = getPosInInt(hashed);

	uint tryCounter = 0;
	uint gotSeat;

	// loop until a bucket is succesfully reserved
	do
	{
		// try to reserve a free bucket
		gotSeat = tryBookASeat(tbIndex, posInInt);

		// check if bucket was already reserved
		if (!gotSeat)
		{
			// rehash the key with the second hash function
			hashed = rehash(hashed, key);

			// Get ticket board index and the position in the integer from hashed value
			tbIndex = getTbIndex(hashed);
			posInInt = getPosInInt(hashed);
		}

		// if the number of tries is bigger than the table size, the table is already full
		assert(++tryCounter < tableSize);
		//throw std::runtime_error("INSERT FAILED - Table is full!");
	} while (!gotSeat);

	// Generate Info from key
	uint info = infoHash(key);

	// Insert info into the reserved bucket
	insertTicketInfo(info, tbIndex, posInInt);

	// Insert key value pair into the table
	keys[hashed] = key;
	values[hashed] = value;

	return hashed;
}

// searches fo a key in the table
template<typename keyT, typename valueT>
__device__ uint StadiumHash<keyT, valueT>::find(keyT key, keyT* keys)
{
	// Intial Hash
	uint hashed = initHash(key);

	// Generate Info from key
	uint info = infoHash(key);


	uint tryCounter = 0;
	uint seatFound;

	// loop until the entry with the searched key is found
	do {
		uint tbIndex = getTbIndex(hashed);
    	uint posInInt = getPosInInt(hashed);

		// search for the key in the ticket board and table
		seatFound = tryFindTheSeat(key, hashed, tbIndex, posInInt, info, keys);

		// if no entry entry was found, rehash the key
		if (!seatFound)
			hashed = rehash(hashed, key);

		// if the number of tries is bigger than the table size, the key isn't included in the table
		if (++tryCounter == tableSize)
			seatFound = KEY_NOT_INSIDE_TABLE;

	} while (!seatFound && seatFound != KEY_NOT_INSIDE_TABLE);


	if (seatFound == KEY_NOT_INSIDE_TABLE)
	{
		// if key wasn't found in the table return tableSize as index
		hashed = tableSize;
	}

	return hashed;
}


template class StadiumHash<uint32_t, uint32_t>;
template class StadiumHash<uint64_t, uint32_t>;
