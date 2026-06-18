#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <matplotlibcpp.h>
#include "plot.h"
#include "linear_regression.h"
#include "data.h"


namespace plt = matplotlibcpp; 

const std::string filename =
std::string(DATASET_DIR) + "/song_data.csv";

int main() {

	try {

	_putenv_s("QT_API", "pyside6"); 

	plt::backend("QtAgg"); 

	const int CAP = 100; 
	const int MIN_RANGE = -100; 
	const int MAX_RANGE = 100; 

	Plot plotGenerator; 
	std::vector<Coords> coords = plotGenerator.generateCoords(CAP, MIN_RANGE, MAX_RANGE); 


	std::vector<double> x_data; 
	std::vector<double> y_data; 
	x_data.reserve(coords.size()); 
	y_data.reserve(coords.size()); 

	for (const auto& c : coords)
	{
		// std::cout << "(x,y) : " << c.x << "," << c.y << std::endl; 
		x_data.push_back(c.x); 
		y_data.push_back(c.y); 
	}


	SimpleLinearRegression slr;
	slr.fit(x_data, y_data); 
	auto y_pred = slr.predict(x_data); 
	
	for (const auto& p : y_pred) {

		std::cout << "y = " << p << '\n'; 
	}

	auto MSE = slr.MSE(y_data, y_pred); 

	std::cout << "MSE: " << MSE << '\n'; 


	DataLoader data; 

	auto dataset = data.parseCSV(filename, true); 

	// print dataset
	for (const auto& row : dataset.features) {

		for (const auto& value : row) {
			std::cout << value << " ";
		}
		std::cout << '\n';
	}



	//MultipleLinearRegression mlr; 




	std::cout << std::filesystem::current_path() << std::endl; 



	/*std::cout << "--SML EQUATION--\n";
	std::cout << "y = " << lr.getSlope() << "x + " << lr.getIntercept() << '\n'; 

	std::vector<double> line_x_coords{ MIN_RANGE, MAX_RANGE }; 


	double y_min = (lr.getSlope() * MIN_RANGE) + lr.getIntercept(); 
	double y_max = (lr.getSlope() * MAX_RANGE) + lr.getIntercept(); 
	std::vector<double> line_y_coords{ y_min, y_max }; 



	plt::xlim(-100, 100);
	plt::ylim(-100, 100);
	plt::title("Linear Regression");
	plt::plot(x_data, y_data, "bo");
	plt::plot(line_x_coords, line_y_coords, "-r"); 
	plt::grid(true);
	plt::show();
	*/

	
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