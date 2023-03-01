#include "DoubleHash.h"

#include <stdexcept>

//
// Generic hash function
template<typename keyT>
template<uint nRounds, uint rShift, uint mulOp>
uint DoubleHash<keyT>::hash(const keyT key)
{
    keyT x = key;
    for (uint j = nRounds; j > 0; --j) {
        x = ((x >> rShift) ^ x) * mulOp + j;
    }
    return (uint)x;
}

//
// Initial hash function
template<typename keyT>
uint DoubleHash<keyT>::initHash(keyT key)
{
    uint hashed = hash<sizeof(keyT), 8, 0xE9D58A6B>(key);
    return hashed % tableSize;
}

//
// Second hash function
template<typename keyT>
uint DoubleHash<keyT>::rehash(uint hashed, const keyT key)
{
    uint h_2 = hash<sizeof(keyT), 8, 0x6E5B9D8A>(key);
    uint dh = hashed + 1 + h_2 % (tableSize - 1);
    return (dh >= tableSize) ? (dh - tableSize) : dh;
}

//
// Initial hash function for arrays
template<typename keyT>
uint DoubleHash<keyT>::arrayHash(keyT* arr, uint length)
{
    uint hashed = 0;
    for (uint i = 0; i < length; i++)
    {
        hashed = hashed * 31 + hash<sizeof(keyT), 8, 0xE9D58A6B>(arr[i]);
    }

    return hashed % tableSize;
}

//
// Second hash function for arrays
template<typename keyT>
uint DoubleHash<keyT>::arrayRehash(uint hashed, keyT* arr, uint length)
{
    uint h_2 = 0;
    for (uint i = 0; i < length; i++)
    {
        h_2 = h_2 * 31 + hash<sizeof(keyT), 8, 0x6E5B9D8A>(arr[i]);
    }

    uint dh = hashed + 1 + h_2 % (tableSize-1);
    return (dh >= tableSize) ? (dh - tableSize) : dh;
}

// Check if a number is a prime number
template<typename keyT>
bool DoubleHash<keyT>::isPrime(int number)
{
    if (number == 2 || number == 3)
        return true;

    if (number % 2 == 0 || number % 3 == 0)
        return false;

    unsigned int divisor = 6;
    while (divisor * divisor - 2 * divisor + 1 <= number)
    {
        if (number % (divisor - 1) == 0)
            return false;

        if (number % (divisor + 1) == 0)
            return false;

        divisor += 6;
    }

    return true;
}

//
// Constructor
template<typename keyT>
DoubleHash<keyT>::DoubleHash(int tableSize)
{
    // Check if table size is a prime number
    if (!isPrime(tableSize))
    {
        throw std::runtime_error("Table size has to be a prime number!\n");
    }

    this->tableSize = tableSize;
}

//
// Find a key in an array of keys of size tableSize
template<typename keyT>
std::pair<bool, int> DoubleHash<keyT>::find(keyT key, keyT* keys)
{
    // intial Hash
    uint hashed = initHash(key);

    std::pair<bool, int> got;
    uint tryCounter = 0;
    uint seatFound = 0;

    // loop until the entry with the searched key or an empty entry is found
    while(!seatFound) 
    {
        // compare the given key and the key at the hashed position
        if (keys[hashed] == 0)
        {
            // if the key is 0, no key was inserted at this position
            seatFound = KEY_NOT_INSIDE_TABLE;
        }
        else if (keys[hashed] == key)
        {
            // if both keys match, the correct key was found
            seatFound = ~0;
        }
        else
        {
            // if the key is non 0 but doesn't match, another key was inserted at this position
            // rehash the key to get the next position
            hashed = rehash(hashed, key);
        }

        // if the number of tries is bigger than the table size, the table is already full and the given key isn't included
        if (++tryCounter == tableSize)
        {
            seatFound = KEY_NOT_INSIDE_TABLE;
            hashed = tableSize;
        }
    }

    
    if (seatFound == KEY_NOT_INSIDE_TABLE)
    {
        // if key wasn't found in the table set first pair member false
        got.first = false;
    }
    else
    {
        // if key was found in the table set first pair member true
        got.first = true;
    }

    // set the last hash of the key as the second pair member
    got.second = hashed;

    return got;
}

template<typename keyT>
std::pair<bool, int> DoubleHash<keyT>::findArray(keyT* key, keyT* keys, uint keyLength)
{
    // intial Hash
    uint hashed = arrayHash(key, keyLength);

    std::pair<bool, int> got;
    uint tryCounter = 0;
    uint zeroCounter = 0;
    uint matchCounter = 0;
    uint seatFound = 0;

    // loop until the entry with the searched key or an empty entry is found
    while(!seatFound) 
    {
        // compare the given key and the key at the hashed position
        zeroCounter = 0;
        matchCounter = 0;
        for (int i = 0; i < keyLength; i++)
        {
            if (keys[(hashed * keyLength) + i] == key[i])
            {
                matchCounter++;
            }

            if (keys[(hashed * keyLength) + i] == 0)
            {
                zeroCounter++;
            }
        }

        if (zeroCounter == keyLength)
        {
            // if the key only contains 0s, no key was inserted at this position
            seatFound = KEY_NOT_INSIDE_TABLE;
        }
        else if (matchCounter == keyLength)
        {
            // if both keys match, the correct key was found
            seatFound = ~0;
        }
        else
        {
            // if the key is non 0 but doesn't match, another key was inserted at this position
            // rehash the key to get the next position
            hashed = arrayRehash(hashed, key, keyLength);
        }   

        // if the number of tries is bigger than the table size, the table is already full and the given key isn't included
        if (++tryCounter == tableSize)
        {
            seatFound = KEY_NOT_INSIDE_TABLE;
            hashed = tableSize;
        }
    }

    if (seatFound == KEY_NOT_INSIDE_TABLE)
    {
        // if key wasn't found in the table set first pair member false
        got.first = false;
    }
    else
    {
        // if key was found in the table set first pair member true
        got.first = true;
    }

    // set the last hash of the key as the second pair member
    got.second = hashed;

    return got;
}

template class DoubleHash<uint64_t>;
template class DoubleHash<uint>;
