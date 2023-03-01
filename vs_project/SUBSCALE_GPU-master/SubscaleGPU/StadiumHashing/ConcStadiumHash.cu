#include "ConcStadiumHash.cuh"
#include <stdint.h>

#define TICKET_SIZE 4
#define NUM_POS_BITS 3 // log(32/TICKET_SIZE)

// Generic Hash Function
template<typename keyT, typename valueT>
template <uint nRounds, uint rShift, uint mulOp>
__device__ uint ConcStadiumHash<keyT, valueT>::hash(const keyT key)
{
    keyT x = key;
    for (uint j = nRounds; j > 0; --j) {
        x = ((x >> rShift) ^ x) * mulOp + j;
    }
    return (uint)x;
}

// First Hash 
template<typename keyT, typename valueT>
__device__ uint ConcStadiumHash<keyT, valueT>::initHash(keyT key)
{
	uint hashed = hash<sizeof(keyT), 8, 0xE9D58A6B>(key);
	return __umulhi(hashed, tableSize);
}

// Second Hash
template<typename keyT, typename valueT>
__device__ uint ConcStadiumHash<keyT, valueT>::rehash(uint hashed, const keyT key)
{
	uint h_2 = hash<sizeof(keyT), 8, 0x6E5B9D8A>(key);
	uint dh = hashed + 1 + __umulhi(h_2, tableSize - 1);
	return (dh >= tableSize) ? (dh - tableSize) : dh;
}

// Info Hash
template<typename keyT, typename valueT>
__device__ uint ConcStadiumHash<keyT, valueT>::infoHash(keyT key)
{
	return hash<sizeof(keyT), 8, 0xCA34BE7D>(key) >> (32 - (TICKET_SIZE - 2)); // mulOp was chosen randomly
}

// First Array Hash
template<typename keyT, typename valueT>
__device__ uint ConcStadiumHash<keyT, valueT>::arrayHash(keyT* arr, uint length)
{
    uint hashed = 0;
    for (uint i = 0; i < length; i++)
    {
        hashed = hashed * 31 + hash<sizeof(keyT), 8, 0xE9D58A6B>(arr[i]);
    }

    return __umulhi(hashed, tableSize);
}

// Second Array Hash
template<typename keyT, typename valueT>
__device__ uint ConcStadiumHash<keyT, valueT>::arrayRehash(uint hashed, keyT* arr, uint length)
{
    uint h_2 = 0;
    for (uint i = 0; i < length; i++)
    {
        h_2 = h_2 * 31 + hash<sizeof(keyT), 8, 0x6E5B9D8A>(arr[i]);
    }

    uint dh = hashed + 1 + __umulhi(h_2, tableSize - 1);
    return (dh >= tableSize) ? (dh - tableSize) : dh;
}

// Info Array Hash
template<typename keyT, typename valueT>
__device__ uint ConcStadiumHash<keyT, valueT>::arrayInfoHash(keyT* arr, uint length)
{
    uint hashed = 0;
    for (uint i = 0; i < length; i++)
    {
        hashed = hashed * 31 + hash<sizeof(keyT), 8, 0xCA34BE7D>(arr[i]);
    }

    return hashed >> (32 - (TICKET_SIZE - 2));
}

// Allocation of the ticket board
template<typename keyT, typename valueT>
__host__ void ConcStadiumHash<keyT, valueT>::allocTicketBoard()
{
    const int ticketBoardSize = tableSize / (32 / TICKET_SIZE) + 1;
    cudaError_t cudaStatus;

    // Allocate memory for ticket board
    cudaStatus = cudaMalloc((void**)&ticketBoard, ticketBoardSize * sizeof(uint));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Ticket Board: cudaMalloc failed!\n");
    }

    // Initialize ticket board with 1s
    clearTicketBoard();
}

// Clearing of the ticket board
template<typename keyT, typename valueT>
__host__ void ConcStadiumHash<keyT, valueT>::clearTicketBoard()
{
    const int ticketBoardSize = tableSize / (32 / TICKET_SIZE) + 1;
    cudaError_t cudaStatus;

    cudaStatus = cudaMemset((void*)ticketBoard, 0xFF, ticketBoardSize * sizeof(uint));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Ticket Board: cudaMemset failed!\n");
    }
}

// Deletion of the ticket board
template<typename keyT, typename valueT>
__host__ void ConcStadiumHash<keyT, valueT>::freeTicketBoard()
{
    cudaFree((void*)ticketBoard);
}

// Creates a mask containing info starting at infoStart and otherwise 1s
template<typename keyT, typename valueT>
__device__ uint ConcStadiumHash<keyT, valueT>::prepareTicket(uint info, uint infoStart)
{
    uint mask = (1 << (TICKET_SIZE - 2)) - 1;
    mask = ~(mask << infoStart);

    uint result = mask | (info << infoStart);

    return result;
}

// Extracts the info from a ticket
template<typename keyT, typename valueT>
__device__ uint ConcStadiumHash<keyT, valueT>::extractInfo(uint ticket, uint infoStart)
{
    uint mask = (1 << (TICKET_SIZE - 2)) - 1;
    mask = mask << infoStart;
    uint result = (mask & ticket) >> infoStart;

    return result;
}

// Calculates Ticket Board Index from the hashed key
template<typename keyT, typename valueT>
__device__ uint ConcStadiumHash<keyT, valueT>::getTbIndex(uint hashed)
{
    return hashed >> NUM_POS_BITS;
}

// Calculates the Position of the ticket in the Ticket Board Entry from the hashed key
template<typename keyT, typename valueT>
__device__ uint ConcStadiumHash<keyT, valueT>::getPosInInt(uint hashed)
{
    uint mask = (1 << NUM_POS_BITS) - 1;

    return (hashed & mask) << (5 - NUM_POS_BITS);
}

// Entry Reservation
template<typename keyT, typename valueT>
__device__ uint ConcStadiumHash<keyT, valueT>::tryBookASeat(uint tbIndex, uint posInInt)
{
    uint availability = 1 << posInInt;
    uint access = 1 << (posInInt + 1);
    uint permit = availability | access;

    // check if availability bit is set
    if (availability & ticketBoard[tbIndex])
    {
        // try to reserve ticket
        uint auth = atomicAnd((uint*)(ticketBoard + tbIndex), ~permit);

        if (auth & availability)
        {
            // ticket was booked succesfully
            return (~0);
        }
        else if (auth & permit == 0)
        {
            // ticket was booked first by another thread
            return 0;
        }
        else
        {
            // ticket is accessed by another thread
            // set access bit (necessary if bit was unset without obtaining the ticket)
            atomicOr((uint*)(ticketBoard + tbIndex), access);
            return 0;
        }
    }
    else
    {
        return 0;
    }
}



// Entry Search
template<typename keyT, typename valueT>
__device__ uint ConcStadiumHash<keyT, valueT>::tryFindTheSeat(keyT* key, uint hashed, uint tbIndex, uint posInInt, uint info, keyT* keys, uint keyLength)
{
    uint permit = 1 << posInInt;
    if (permit & ticketBoard[tbIndex]) {
        return KEY_NOT_INSIDE_TABLE;
    }
    else {

        // wait until ticket can be accessed
        while (isAccessed(tbIndex, posInInt)) {};

        // get and compare info from ticket
        uint retrievedInfo = extractInfo(ticketBoard[tbIndex], posInInt + 2);
        if (info != retrievedInfo) return 0;

        // compare keys
        for (int i = 0; i < keyLength; i++)
        {
            if (keys[i] != key[i]) return 0;
        }
        
        return (~0);
    }
}

// Entry Search
template<typename keyT, typename valueT>
__device__ uint ConcStadiumHash<keyT, valueT>::tryFindTheSeat(keyT key, uint hashed, uint tbIndex, uint posInInt, uint info, keyT* keys)
{
    uint permit = 1 << posInInt;
    if (permit & ticketBoard[tbIndex]) {
        return KEY_NOT_INSIDE_TABLE;
    }
    else {
        // wait until ticket can be accessed
        while (isAccessed(tbIndex, posInInt)) {};

        // get info from ticket
        uint retrievedInfo = extractInfo(ticketBoard[tbIndex], posInInt + 2); // ACCESS CHANGE
        if (info != retrievedInfo) return 0;

        // compare keys
        if (key != keys[hashed]) return 0;

        return (~0);
    }
}

// Inserts info into the ticket board
template<typename keyT, typename valueT>
__device__ void ConcStadiumHash<keyT, valueT>::insertTicketInfo(uint info, uint tbIndex, uint posInInt)
{
    uint prepTicket = prepareTicket(info, posInInt + 2);
    atomicAnd((uint*)ticketBoard + tbIndex, prepTicket);
}

// Checks if a ticket is accessed by a thread
template<typename keyT, typename valueT>
__device__ uint ConcStadiumHash<keyT, valueT>::isAccessed(uint tbIndex, uint posInInt)
{
    uint access = (1 << (posInInt + 1));

    return (ticketBoard[tbIndex] & access) ? (0) : (~0);
}

// Unlocks access to the ticket
template<typename keyT, typename valueT>
__device__ void ConcStadiumHash<keyT, valueT>::unlockAccess(uint tbIndex, uint posInInt)
{
    __threadfence();
    uint access = (1 << (posInInt + 1));
    atomicOr((uint*)ticketBoard + tbIndex, access);
}

// Constructor
template<typename keyT, typename valueT>
ConcStadiumHash<keyT, valueT>::ConcStadiumHash(uint tableSize)
{
    this->tableSize = tableSize;
}

// Destructor
template<typename keyT, typename valueT>
ConcStadiumHash<keyT, valueT>::~ConcStadiumHash()
{
    freeTicketBoard();
}

// Inserts a given key value pair into the table
template<typename keyT, typename valueT>
__device__ void ConcStadiumHash<keyT, valueT>::insert(keyT* key, valueT value, keyT* keys, valueT* values, uint keyLength)
{
    // Initial Hash
    uint hashed = arrayHash(key, keyLength);

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
            hashed = arrayRehash(hashed, key, keyLength);

            // Get ticket board index and the position in the integer from hashed value
            tbIndex = getTbIndex(hashed);
            posInInt = getPosInInt(hashed);
        }

        // if the number of tries is bigger than the table size, the table is already full
        assert(++tryCounter < tableSize);
        //throw std::runtime_error("INSERT FAILED - Table is full!");
    } while (!gotSeat);

    // Generate Info from key
    uint info = arrayInfoHash(key, keyLength);

    // Insert info into the reserved bucket
    insertTicketInfo(info, tbIndex, posInInt);

    // Insert key value pair into the table
    for (int i = 0; i < keyLength; i++)
    {
        keys[hashed * keyLength + i] = key[i];
    }

    values[hashed] = value;

    unlockAccess(tbIndex, posInInt);
}

// searches fo a key in the table
template<typename keyT, typename valueT>
__device__ uint ConcStadiumHash<keyT, valueT>::find(keyT* key, keyT* keys, uint keyLength)
{
    // Intial Hash
    uint hashed = arrayHash(key, keyLength); // initHash(key);
    // printf("Key: %u Hashed: %u\n", key, hashed);
    // Generate Info from key
    uint info = arrayInfoHash(key, keyLength);

    uint tbIndex = getTbIndex(hashed);
    uint posInInt = getPosInInt(hashed);

    uint tryCounter = 0;
    uint seatFound;

    // loop until the entry with the searched key is found
    do {
        // search for the key in the ticket board and table
        seatFound = tryFindTheSeat(key, hashed, tbIndex, posInInt, info, keys + (hashed*keyLength), keyLength);

        // if no entry entry was found, rehash the key
        if (!seatFound)
        {
            hashed = arrayRehash(hashed, key, keyLength);
            tbIndex = getTbIndex(hashed);
            posInInt = getPosInInt(hashed);
        }

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

template class ConcStadiumHash<unsigned int, unsigned int>;
template class ConcStadiumHash<uint64_t, uint32_t>;
