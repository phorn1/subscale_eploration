#pragma once
#include <vector>
#include <list>
#include "../SubscaleTypes.h"

using namespace std;

// class for creating core sets
class CoreSetCreator
{
private:
	vector<CoreSet> createCoreSetsInDimension(vector<PointIDValue> dimensionVector, int minPoints, double epsilon, int numberOfPoints);
	int getPivotIndex(list<unsigned int> ids, unsigned int previousLastID);
public:
	vector<vector<CoreSet>> createCoreSets(vector<DataPoint> points, int minPoints, double epsilon);
};

