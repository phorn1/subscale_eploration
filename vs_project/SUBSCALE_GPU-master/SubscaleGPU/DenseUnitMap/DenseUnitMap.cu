#include "DenseUnitMap.cuh"

// insert dense unit into the hash map
__device__ void DenseUnitMap::insertEntry(uint64_t signature, uint32_t* ids, uint32_t dimension)
{
    // Intial Hash
    uint hashed = initHash(signature);

    // Generate Info from key
    uint info = infoHash(signature);

    uint tbIndex = getTbIndex(hashed);
    uint posInInt = getPosInInt(hashed);

    uint gotSeat;

    uint tryCounter = 0;
    uint seatFound;

    // loop until the dense unit is succesfully inserted
    do {
        // loop until the entry with the searched key is found
        do {

            // search for the key in the ticket board and table
            seatFound = tryFindTheSeat(signature, hashed, tbIndex, posInInt, info, keys);

            // if no entry entry was found, rehash the key
            if (!seatFound)
            {
                hashed = rehash(hashed, signature);

                tbIndex = getTbIndex(hashed);
                posInInt = getPosInInt(hashed);
            }

            // if the number of tries is bigger than the table size, the key isn't included in the table
            if (++tryCounter >= tableSize)
            {
                // assert(false);
                __threadfence();
                asm("trap;"); // stop threads
            }

        } while (!seatFound && seatFound != KEY_NOT_INSIDE_TABLE);


        if (seatFound == KEY_NOT_INSIDE_TABLE)
        {
            // If there is no dense unit with the given signature, the given dense unit is inserted into the table
            gotSeat = tryBookASeat(tbIndex, posInInt);

            if (gotSeat)
            {
                insertTicketInfo(info, tbIndex, posInInt);

                keys[hashed] = signature;

                table->insertIds(ids, hashed);
                table->dev_addDimension(dimension, hashed);

            }
        }
        else
        {
            gotSeat = ~0;

            // if there is already a dense unit with the given signature, the given dimensionen is added to the entry
            table->dev_addDimension(dimension, hashed);

        }
    } while (!gotSeat);
}


// Constructor
DenseUnitMap::DenseUnitMap(DenseUnitTable* table, uint32_t tableSize, uint32_t ticketSize)
    : StadiumHash(tableSize, ticketSize)
{
    this->table = table;
}

