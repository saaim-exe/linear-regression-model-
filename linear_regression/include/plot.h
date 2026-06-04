#pragma once
#include <vector> 
#include <random>

struct Coords {
	double x;
	double y; 
};

class Plot {

public:

	Plot() = default; 

	std::vector<Coords> generateCoords(size_t CAP = 100, int min = -100, int max = 100); 

private: 

};

