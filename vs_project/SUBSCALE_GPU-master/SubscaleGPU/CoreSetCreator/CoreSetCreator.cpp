#include "CoreSetCreator.h"
#include <algorithm>

// find all core sets in one dimension
vector<CoreSet> CoreSetCreator::createCoreSetsInDimension(vector<PointIDValue> dimensionVector, int minPoints, double epsilon, int numberOfPoints)
{
	vector<CoreSet> coreSetVector;
	CoreSet coreSet;
    
    std::list<unsigned int> ids;

    int previousLastID = -1;
    int iteratingIndex = 0;


    // Iterate once across all point values in one dimension vector
    for (PointIDValue pointIDValue : dimensionVector) {

        // Iterate from the next point onwards until the distance to the current point gets bigger than epsilon and add the next point ID to the core set
        while ((iteratingIndex < numberOfPoints) && (dimensionVector[iteratingIndex].value - pointIDValue.value) <= epsilon) {
            // coreSet.ids.push_back(dimensionVector[iteratingIndex].id);
            ids.push_back(dimensionVector[iteratingIndex].id);
            iteratingIndex++;
        }

        // Check whether the last ID of the core set stayed the same as in the round before
        int currentLastID = ids.back(); // set.get(set.size() - 1);
        if (currentLastID != previousLastID) 
        {
            // std::find(coreSet.ids.begin(), coreSet.ids.end(), previousLastID);
            coreSet.pivot = getPivotIndex(ids, previousLastID);

            coreSet.size = ids.size();
            //Check whether each point of this set has at least TAU neighbours,
            //    => The current point has already been added to the set and therefore does not add to count
            if (coreSet.size >= minPoints) {
                // set.trimToSize();
                // densityChunks.add(new ArrayList<>(set));
                
                coreSet.ids = new unsigned int[coreSet.size];
                std::copy(ids.begin(), ids.end(), coreSet.ids);
                coreSetVector.push_back(coreSet);

                // Set last rounds last point ID to the ID of the currently last point of the set
                previousLastID = currentLastID;
            }
        }

        // Remove the first element of the set to move to the density chunk starting with next element
        // coreSet.ids.pop_front();
        ids.pop_front();
    }

    coreSetVector.shrink_to_fit();
	return coreSetVector;
}

// searches for last id of the previous core set in the given ids of the current core set
int CoreSetCreator::getPivotIndex(list<unsigned int> ids, unsigned int previousLastID)
{
    int pivotIndex = 0;
    for (unsigned int id : ids)
    {
        if (id == previousLastID)
        {
            return pivotIndex;
        }

        pivotIndex++;
    }

    return -1;
}

// creates all core sets in all dimensions
vector<vector<CoreSet>> CoreSetCreator::createCoreSets(vector<DataPoint> points, int minPoints, double epsilon)
{
	vector<vector<CoreSet>> allCoreSets;

	int numberOfDimensions = points[0].values.size();
	int numberOfPoints = points.size();

	// PointIDValue* dimensionVector = new PointIDValue[numberOfPoints];
	vector<PointIDValue> dimensionVector(numberOfPoints);

	for (int dimension = 0; dimension < points[0].values.size(); dimension++)
	{
		
		for (int i = 0; i < numberOfPoints; i++)
		{
			dimensionVector[i].id = points[i].id;
			dimensionVector[i].value = points[i].values[dimension];
		}

		std::sort(dimensionVector.begin(), dimensionVector.end());

		allCoreSets.push_back(createCoreSetsInDimension(dimensionVector, minPoints, epsilon, numberOfPoints));
	}

    allCoreSets.shrink_to_fit();

	return allCoreSets;
}

