#include "CsvDataHandler.h"


// reads a file and
vector<DataPoint> CsvDataHandler::read(const char* path, char delimiter)
{
	// Create an input filestream
	std::ifstream inputFile(path);

	// Make sure the file is open
	if (!inputFile.is_open()) throw std::runtime_error("Could not open file");


	// Vector of points
	vector<DataPoint> points;

	double val;
	std::string line;
	unsigned int rowIndex = 0;


	// Read data, line by line
	while (std::getline(inputFile, line))
	{
		// Create a new point
		DataPoint newPoint;

		// Create a stringstream of the current line
		std::stringstream ss(line);

		// Set id of the point to the index of the row
		newPoint.id = rowIndex;

		// Extract each value from the current line
		while (ss >> val)
		{
			// Add the value to the point
			newPoint.values.push_back(val);

			// If the next token is the delimter, ignore it and move on
			if (ss.peek() == delimiter) ss.ignore();
		}

		// Add the current point to the vector of points
		points.push_back(newPoint);

		// Increase the index of the row
		rowIndex++;
	}

	// Close input filestream
	inputFile.close();

	return points;
}

// writes a subscale table to a file at the given path
void CsvDataHandler::writeTable(const char* path, SubscaleTable* table, unsigned int numberOfEntries)
{
	// Create an output filestream object
	std::ofstream myFile(path);

	unsigned int bufferIndex = 0;
	int len;

	// get table sizes
	int tableSize = table->getTableSize();
	int dimensionsSize = table->getDimensionsSize();
	int idsSize = table->getIdsSize();

	for (int i = 0; i < numberOfEntries; i++)
	{
		unsigned int* dimensions = table->getDimensions(i);
		unsigned int* ids = table->getIds(i);

		buffer[bufferIndex++] = '[';

		// Dimensions
		for (int j = 0; j < dimensionsSize - 1; j++)
		{
			len = sprintf(buffer + bufferIndex, "%u", dimensions[j]);
			bufferIndex += len;
			buffer[bufferIndex++] = ',';
		}

		// last dimension value (isn't followed by a comma)
		len = sprintf(buffer + bufferIndex, "%u", dimensions[dimensionsSize - 1]);
		bufferIndex += len;

		// closing bracket for dimensions and opening bracket for ids
		buffer[bufferIndex++] = ']';
		buffer[bufferIndex++] = '-';
		buffer[bufferIndex++] = '[';


		// IDs
		for (int j = 0; j < idsSize - 1; j++)
		{
			len = sprintf(buffer + bufferIndex, "%u", ids[j]);
			bufferIndex += len;
			buffer[bufferIndex++] = ',';
		}

		// last id value (isn't followed by a comma)
		len = sprintf(buffer + bufferIndex, "%u", ids[idsSize - 1]);
		bufferIndex += len;

		// closing bracket for ids and start new line
		buffer[bufferIndex++] = ']';
		buffer[bufferIndex++] = '\n';

	}

	myFile.write(buffer, bufferIndex);

	// Close the file
	myFile.close();
}

// writes a subspace table to a file at the given path and converts bitmaps to vectors
void CsvDataHandler::writeVecTable(const char* path, SubspaceTable* table, unsigned int numberOfEntries)
{
	// Create an output filestream object
	std::ofstream myFile(path);

	unsigned int bufferIndex = 0;
	int len;

	// get table sizes
	int tableSize = table->getTableSize();

	for (int i = 0; i < numberOfEntries; i++)
	{
		vector<unsigned int> dimensionsVec = table->getDimensionsVec(i);
		vector<unsigned int> idsVec = table->getIdsVec(i);
		

		buffer[bufferIndex++] = '[';

		// Dimensions
		for (int j = 0; j < dimensionsVec.size() - 1; j++)
		{
			len = sprintf(buffer + bufferIndex, "%u", dimensionsVec[j]);
			bufferIndex += len;
			buffer[bufferIndex++] = ',';
		}

		// last dimension value (isn't followed by a comma)
		len = sprintf(buffer + bufferIndex, "%u", dimensionsVec[dimensionsVec.size() - 1]);
		bufferIndex += len;

		// closing bracket for dimensions and opening bracket for ids
		buffer[bufferIndex++] = ']';
		buffer[bufferIndex++] = '-';
		buffer[bufferIndex++] = '[';


		// IDs
		for (int j = 0; j < idsVec.size() - 1; j++)
		{
			len = sprintf(buffer + bufferIndex, "%u", idsVec[j]);
			bufferIndex += len;
			buffer[bufferIndex++] = ',';
		}

		// last id value (isn't followed by a comma)
		len = sprintf(buffer + bufferIndex, "%u", idsVec[idsVec.size() - 1]);
		bufferIndex += len;

		// closing bracket for ids and start new line
		buffer[bufferIndex++] = ']';
		buffer[bufferIndex++] = '\n';

	}

	myFile.write(buffer, bufferIndex);

	// Close the file
	myFile.close();
}

// writes clusters to a file
void CsvDataHandler::writeClusters(const char* path, vector<Cluster> clusters)
{
	// Create an output filestream object
	std::ofstream myFile(path);

	unsigned int bufferIndex = 0;
	int len;


	for (Cluster cluster : clusters)
	{
		vector<unsigned int> dimensionsVec = cluster.subspace;
		vector<unsigned int> idsVec = cluster.ids;

		buffer[bufferIndex++] = '[';

		// Dimensions
		for (int j = 0; j < dimensionsVec.size() - 1; j++)
		{
			len = sprintf(buffer + bufferIndex, "%u", dimensionsVec[j]);
			bufferIndex += len;
			buffer[bufferIndex++] = ',';
		}

		// last dimension value (isn't followed by a comma)
		len = sprintf(buffer + bufferIndex, "%u", dimensionsVec[dimensionsVec.size() - 1]);
		bufferIndex += len;

		// closing bracket for dimensions and opening bracket for ids
		buffer[bufferIndex++] = ']';
		buffer[bufferIndex++] = '-';
		buffer[bufferIndex++] = '[';


		// IDs
		for (int j = 0; j < idsVec.size() - 1; j++)
		{
			len = sprintf(buffer + bufferIndex, "%u", idsVec[j]);
			bufferIndex += len;
			buffer[bufferIndex++] = ',';
		}

		// last id value (isn't followed by a comma)
		len = sprintf(buffer + bufferIndex, "%u", idsVec[idsVec.size() - 1]);
		bufferIndex += len;

		// closing bracket for ids and start new line
		buffer[bufferIndex++] = ']';
		buffer[bufferIndex++] = '\n';

	}

	myFile.write(buffer, bufferIndex);

	// Close the file
	myFile.close();
}


// reads subspace table from a file
unsigned int CsvDataHandler::readTable(const char* path, SubspaceTable* table)
{

	std::ifstream inputFile(path);

	// Make sure the file is open
	if (!inputFile.is_open()) throw std::runtime_error("Could not open file");

	unsigned int rowIndex = 0;
	unsigned int* entry;

	inputFile.read(buffer, bufferSize);

	int remainingChars = inputFile.gcount();

	unsigned int index = 0;

	for (int i = 0; i < remainingChars; i++)
	{
		switch (buffer[i])
		{
		case ',':
			// Ignore character and switch to the next field of the entry
			i++;
			index++;
			break;

		case '[':
			// Ignore character and start with a new entry (only occurs once, at the first character)
			entry = table->getDimensions(rowIndex);
			i++;
			index = 0;
			break;

		case ']':
			if (buffer[i + 1] == '\n')
			{
				// Ignore next 3 characters (]\n[) and start with a new entry
				rowIndex++;
				entry = table->getDimensions(rowIndex);
				i += 3;
			}
			else
			{
				// Ignore next 3 characters (]-[) and switch do the ids of the entry
				entry = table->getIds(rowIndex);
				i += 3;
			}

			index = 0;
			break;

		}

		if (i < remainingChars)
		{
			// add character to the entry
			// If the number is larger than 1 character, all characters have to be added to the same entry field.
			// All entries have to be 0 at the start for this to work.
			entry[index] = entry[index] * 10 + buffer[i] - '0';
		}
	}

	// Close input filestream
	inputFile.close();

	return rowIndex;
}


