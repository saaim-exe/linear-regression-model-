#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <map>
#include <matplotlibcpp.h>
#include "plot.h"

namespace plt = matplotlibcpp; 

 
int main() {

	try {

	_putenv_s("QT_API", "pyside6"); 

	plt::backend("QtAgg"); 


	Plot plotGenerator; 
	std::vector<Coords> coords = plotGenerator.generateCoords(100, -100, 100); 


	std::vector<double> x_data; 
	std::vector<double> y_data; 
	x_data.reserve(coords.size()); 
	y_data.reserve(coords.size()); 

	for (const auto& c : coords)
	{
		std::cout << "(x,y) : " << c.x << "," << c.y << std::endl; 
		x_data.push_back(c.x); 
		y_data.push_back(c.y); 
	}

	plt::xlim(-100, 100); 
	plt::ylim(-100, 100); 
	plt::title("Linear Regression"); 
	plt::plot(x_data, y_data, "ro"); 
	plt::grid(true);
	plt::show(); 

	
	

	
	return 0;

	}
	catch (const std::exception& ex)
	{
		std::cerr << "Runtime Error: " << ex.what() << std::endl;
		if (PyErr_Occurred()) {
			PyErr_Print(); 
		}
		return 1;
	}
}