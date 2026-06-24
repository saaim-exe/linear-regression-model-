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


void Plot::LOBF(const std::vector<double>& x_data, const std::vector<double>& y_data, double slope, double bias) const {




}

void Plot::actual_vs_predicted(const VectorXd& y_test, const VectorXd& y_pred) const {
	
	assert(y_test.size() == y_pred.size());

	std::vector<double> x_data(y_pred.data(), y_pred.data() + y_pred.size()); 
	std::vector<double> y_data(y_test.data(), y_test.data() + y_test.size()); 

	double min_val = std::min(y_pred.minCoeff(), y_test.minCoeff()); 
	double max_val = std::max(y_pred.maxCoeff(), y_test.maxCoeff()); 

	std::vector<double> line_x{ min_val, max_val }; 
	std::vector<double> line_y{ min_val, max_val }; 

	
	plt::xlim(min_val, max_val); 
	plt::ylim(min_val, max_val); 
	plt::xlabel("Predicted Values"); 
	plt::ylabel("Actual Values"); 
	plt::title("Actual vs Predicted");
	plt::scatter(x_data, y_data); 
	plt::plot(line_x, line_y, "r--"); 
	plt::grid(true); 

	plt::save(std::string(IMAGES_DIR) + "/actual_vs_predicted.png");
	plt::clf(); 

} 

void Plot::residuals(const VectorXd& y_test, const VectorXd& y_pred) const {

	assert(y_test.size() == y_pred.size());


	VectorXd residuals = y_test - y_pred; 

	std::vector<double> x_data(y_pred.data(), y_pred.data() + y_pred.size()); 
	std::vector<double> y_data(residuals.data(), residuals.data() + residuals.size()); 

	std::vector<double> zero_x{
		y_pred.minCoeff(),
		y_pred.maxCoeff()
	}; 

	std::vector<double> zero_y{ 0.0, 0.0 }; 

	plt::xlim(y_pred.minCoeff(), y_pred.maxCoeff()); 
	plt::ylim(residuals.minCoeff(), residuals.maxCoeff()); 
	plt::xlabel("Predicted Values"); 
	plt::ylabel("Residual Values"); 
	plt::title("Residual Plot"); 
	plt::scatter(x_data, y_data); 
	plt::plot(zero_x, zero_y, "r--"); 
	plt::grid(true);

	plt::save(std::string(IMAGES_DIR) + "/residual_plot.png");
	plt::clf(); 

}

void Plot::print_metrics_closed_form(double mse, double r2, double adj_r2) const {
	
	std::cout << std::fixed << std::setprecision(4);

	std::cout << "\n";
	std::cout << "=============================\n";
	std::cout << "     Closed Form OLS\n";
	std::cout << "=============================\n";
	std::cout << "MSE:          " << mse << '\n';
	std::cout << "R Squared:    " << r2 << '\n';
	std::cout << "Adjusted R Squared:  " << adj_r2 << '\n';
	std::cout << "=============================\n\n";

}

void Plot::print_metrics_gradient_descent(double mse, double r2, double adj_r2) const {

	std::cout << std::fixed << std::setprecision(4);

	std::cout << "\n";
	std::cout << "=============================\n";
	std::cout << "     Gradient Descent OLS\n";
	std::cout << "=============================\n";
	std::cout << "MSE:          " << mse << '\n';
	std::cout << "R Squared:    " << r2 << '\n';
	std::cout << "Adjusted R Squared:  " << adj_r2 << '\n';
	std::cout << "=============================\n\n";

}

