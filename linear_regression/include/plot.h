#pragma once
#include <vector> 
#include <random>
#include <iomanip>
#include "linear_regression.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp; 


struct Coords {
	double x;
	double y; 
};

class Plot {

public:

	Plot() = default; 

	std::vector<Coords> generateCoords(size_t CAP = 100, int min = -100, int max = 100); 
	
	// SLR 
	void LOBF(const std::vector<double>& x_data, const std::vector<double>& y_data, double slope, double bias) const; 
	
	// MLR 
	void actual_vs_predicted(const VectorXd& y_test, const VectorXd& y_pred) const; 
	void residuals(const VectorXd& y_test, const VectorXd& y_pred) const; 

	void print_metrics_closed_form(double mse, double r2, double adj_r2) const; 
	void print_metrics_gradient_descent(double mse, double r2, double adj_r2) const; 

};

