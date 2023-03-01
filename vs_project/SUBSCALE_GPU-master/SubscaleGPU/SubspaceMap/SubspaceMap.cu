#include "SubspaceMap.cuh"
#include "device_launch_parameters.h"

// Constructor
SubspaceMap::SubspaceMap(SubspaceTable* table, uint32_t tableSize, uint32_t ticketSize) 
    :ConcStadiumHash(tableSize)
{
    this->table = table;
}

// Inserts an entry into the table
__device__ void SubspaceMap::insertEntry(uint32_t* ids, uint32_t* dimensions)
{
    uint dimensionsSize = table->getDimensionsSize();

    // Intial Hash
    uint hashed = arrayHash(dimensions, dimensionsSize);

    // Generate Info from key
    uint info = arrayInfoHash(dimensions, dimensionsSize);

    uint tbIndex = getTbIndex(hashed);
    uint posInInt = getPosInInt(hashed);

    uint gotSeat;

    uint tryCounter = 0;
    uint seatFound;

    uint* tableDimensions;

    // loop until the dense unit is succesfully inserted
    do {
        // loop until the entry with the searched key is found
        do {
            // search for the key in the ticket board and table
            tableDimensions = table->getDimensions(hashed);
            
            seatFound = tryFindTheSeat(dimensions, hashed, tbIndex, posInInt, info, tableDimensions, dimensionsSize);

            // if no entry entry was found, rehash the key
            if (!seatFound)
            {
                hashed = arrayRehash(hashed, dimensions, dimensionsSize);

                tbIndex = getTbIndex(hashed);
                posInInt = getPosInInt(hashed);
            }

            // if the number of tries is bigger than the table size, the key isn't included in the table
            if (++tryCounter >= tableSize)
            {
                assert(false);
                __threadfence();
                asm("trap;");
                
            }

        } while (!seatFound && seatFound != KEY_NOT_INSIDE_TABLE);


        if (seatFound == KEY_NOT_INSIDE_TABLE)
        {
            // If there is no dense unit with the given signature, the given dense unit is inserted into the table
            // gotSeat = tryBookASeatWithAccess(tbIndex, posInInt);
            gotSeat = tryBookASeat(tbIndex, posInInt);

            if (gotSeat)
            {
                insertTicketInfo(info, tbIndex, posInInt);

                table->insertDimensions(dimensions, hashed);
                table->dev_mergeIds(ids, hashed);
                
                unlockAccess(tbIndex, posInInt);
            }
        }
        else
        {
            gotSeat = 1;

            // if there is already a dense unit with the given signature, the given dimensionen is added to the entry
            table->dev_mergeIds(ids, hashed);

        }
    } while (!gotSeat);
}


