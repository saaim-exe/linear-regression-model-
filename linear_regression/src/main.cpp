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


//	std::cout << std::filesystem::current_path() << std::endl; 

namespace plt = matplotlibcpp; 

const std::string filename =
std::string(DATASET_DIR) + "/song_data.csv";

int main() {

	try {

	_putenv_s("QT_API", "pyside6"); 

	plt::backend("QtAgg"); 


	// SLR


/*	const int CAP = 100;
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


	*/

	DataLoader loader; 

	auto raw_data = loader.parseCSV(filename, true); 

	// //print dataset
	//for (const auto& row : raw_data.features) {
	//
	//	for (const auto& value : row) {
	//		std::cout << value << " ";
	//	}
	//	std::cout << '\n';
	//}



	// MLR 

	MultipleLinearRegression mlr_model; 
	auto mlr_data = mlr_model.loadData(raw_data); 
	auto mlr_split = mlr_model.train_test_split(mlr_data);
	
	auto stats = mlr_model.fit_normalizer(mlr_split.X_train); 
	auto X_train_norm = mlr_model.normalize(mlr_split.X_train, stats); 
	auto X_test_norm = mlr_model.normalize(mlr_split.X_test, stats);

	auto mlr_params = mlr_model.fit(X_train_norm, mlr_split.y_train); 

	auto y_test_pred = mlr_model.predict(X_test_norm, mlr_params); 

	auto MSE = mlr_model.MSE(y_test_pred, mlr_split.y_test); 

	auto R_sq = mlr_model.R_squared(y_test_pred, mlr_split.y_test); 
	auto adj_R_sq = mlr_model.adjusted_R_squared(X_test_norm, y_test_pred, mlr_split.y_test);

	// gradient descent 

	parameters_mlr mlr_params_gd; 
	mlr_params_gd.weights = VectorXd::Zero(X_train_norm.cols()); 
	mlr_params_gd.bias = 0.0; 

	const int epochs = 10000;
	const double learning_rate = 0.01; 

	for (int i = 0; i < epochs; ++i) {
		mlr_params_gd = mlr_model.gradient_descent(
			X_train_norm,
			mlr_split.y_train,
			mlr_params_gd,
			learning_rate
		); 
	}

	auto y_test_pred_gd = mlr_model.predict(X_test_norm, mlr_params_gd);
	auto MSE_gd = mlr_model.MSE(y_test_pred_gd, mlr_split.y_test);
	auto R_sq_gd = mlr_model.R_squared(y_test_pred_gd, mlr_split.y_test);
	auto adj_R_sq_gd = mlr_model.adjusted_R_squared(X_test_norm, y_test_pred_gd, mlr_split.y_test);



	// PLOTTING 

	Plot p; 
	p.print_metrics_closed_form(MSE, R_sq, adj_R_sq);
	p.print_metrics_gradient_descent(MSE_gd, R_sq_gd, adj_R_sq_gd); 
	p.actual_vs_predicted(mlr_split.y_test, y_test_pred); 
	p.residuals(mlr_split.y_test, y_test_pred); 
	

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
