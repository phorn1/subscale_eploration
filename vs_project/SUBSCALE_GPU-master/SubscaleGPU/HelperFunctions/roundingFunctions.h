#pragma once
//
// Helper functions for rounding numbers

// checks if a number is a prime number
bool isPrime(unsigned int number);

// increases a number until the next prime number is reached
unsigned int roundToNextPrime(unsigned int number);

// calculates the next quotient greater or equal than number that can be divided by divisor
int roundToNextQuotient(int number, int divisor);
