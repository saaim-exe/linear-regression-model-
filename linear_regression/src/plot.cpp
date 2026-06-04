#include "plot.h"


std::vector<Coords> Plot::generateCoords(size_t CAP, int min, int max) {

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