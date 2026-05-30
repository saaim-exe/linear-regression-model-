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

	std::vector<Coords> generateCoords(size_t CAP = 100, int min = -100, int max = 100) {
		
		std::random_device rd; 
		std::mt19937 gen(rd()); 
		std::uniform_int_distribution<> distrib(min, max); 
		
		std::vector<Coords> coordinates; 
		coordinates.reserve(CAP); 

		for (size_t i = 0; i < CAP; ++i)
		{
			coordinates.emplace_back(distrib(gen), distrib(gen)); 
		}

		return coordinates; 
	}



private: 









};

