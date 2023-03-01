#include "roundingFunctions.h"

// checks if a number is a prime number
bool isPrime(unsigned int number)
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

// increases a number until the next prime number is reached
unsigned int roundToNextPrime(unsigned int number)
{
    while (!isPrime(number))
    {
        number++;
    }

    return number;
}

// calculates the next quotient greater or equal than number that can be divided by divisor
int roundToNextQuotient(int number, int divisor)
{
    if (number % divisor != 0)
    {
        number = number / divisor + 1;
    }
    else
    {
        number = number / divisor;
    }

    return number;
}
