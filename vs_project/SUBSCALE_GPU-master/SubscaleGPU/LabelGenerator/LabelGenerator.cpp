#include "LabelGenerator.h"
#include <set>

// Constructor
LabelGenerator::LabelGenerator(unsigned long long minLabel, unsigned long long maxLabel)
{
	gen = std::mt19937_64(rd()); // //Standard mersenne_twister_engine seeded with rd()
	dis = std::uniform_real_distribution<double>(0, 1); // Generator for uniformily distributed doubles between 0 and 1

	this->minLabel = minLabel;
	this->maxLabel = maxLabel;
}

// generates random labels and stores them in the given array
void LabelGenerator::getLabels(unsigned long long* labels, int numberOfLabels)
{
	for (int i = 0; i < numberOfLabels; i++)
	{
		labels[i] = getRandomLabel();
	}
}

// calculates the minimal signature that can result from the given labels
unsigned long long LabelGenerator::calcMinSignature(unsigned long long* labels, int numberOfLabels, int minPoints)
{
	unsigned long long minSignature = 0;
	std::set<unsigned long long> minLabels;

	// fill minLabels with the first minPoints labels
	for (int i = 0; i < minPoints; i++)
	{
		minLabels.insert(labels[i]);
	}

	for (int i = minPoints; i < numberOfLabels; i++)
	{
		// check if label is smaller than the greatest label in minLabels
		if (labels[i] < *(--minLabels.end()))
		{
			// check if minLabels already contains label
			if (minLabels.count(labels[i]) == 0)
			{
				// insert label into minLabels and erase the greatest label
				minLabels.erase(--minLabels.end());
				minLabels.insert(labels[i]);
			}
		}
	}

	// calculate minSignature by accumalating all minLabels
	for (unsigned long long label : minLabels)
	{
		minSignature += label;
	}

	return minSignature;
}

// calculates the maximal signature that can result from the given labels
unsigned long long LabelGenerator::calcMaxSignature(unsigned long long* labels, int numberOfLabels, int minPoints)
{
	unsigned long long maxSignature = 0;
	std::set<unsigned long long> maxLabels;

	// fill maxLabels with the first minPoints labels
	for (int i = 0; i < minPoints; i++)
	{
		maxLabels.insert(labels[i]);
	}

	for (int i = minPoints; i < numberOfLabels; i++)
	{
		// check if label is greater than the smallest label in maxLabels
		if (labels[i] > *(maxLabels.begin()))
		{
			// check if maxLabels already contains label
			if (maxLabels.count(labels[i]) == 0)
			{
				// insert label into maxLabels and erase the smallest label
				maxLabels.erase(maxLabels.begin());
				maxLabels.insert(labels[i]);
			}
		}
	}

	// calculate maxSignature by accumalating all maxLabels
	for (unsigned long long label : maxLabels)
	{
		maxSignature += label;
	}

	return maxSignature;
}

// generates a random label
unsigned long long LabelGenerator::getRandomLabel()
{
	return minLabel + dis(gen) * (maxLabel-minLabel);
}
