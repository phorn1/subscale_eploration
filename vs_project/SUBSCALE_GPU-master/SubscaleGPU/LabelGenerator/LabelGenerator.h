#pragma once
#include <random>

// class for generating labels
class LabelGenerator
{
private:
	std::random_device rd;
	std::mt19937_64 gen;
	std::uniform_real_distribution<double> dis;
	unsigned long long minLabel;
	unsigned long long maxLabel;
public:
	LabelGenerator(unsigned long long minLabel, unsigned long long maxLabel);
	void getLabels(unsigned long long* labels, int numberOfLabels);

	unsigned long long calcMinSignature(unsigned long long* labels, int numberOfLabels, int minPoints);
	unsigned long long calcMaxSignature(unsigned long long* labels, int numberOfLabels, int minPoints);

	unsigned long long getRandomLabel();
};

