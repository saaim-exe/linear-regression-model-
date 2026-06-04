#include "linear_regression.h"


double LinearRegression::mean(const std::vector<double>& values) const{


	if (values.empty()) {
		throw std::invalid_argument("Empty Vector"); 
	}

	const size_t N = values.size(); 
	double sum = 0; 

	for (double val : values)
	{
		sum += val; 
	}

	return sum / N; 
}

double LinearRegression::variance(const std::vector<double>& x) const {

	return covariance(x, x); 
}

double LinearRegression::covariance(const std::vector<double>& x, const std::vector<double>& y) const {

	assert(!x.empty() && "X Vector is Empty!"); 
	assert(x.size() == y.size() && "Vector sizes do not match!"); 
	
	double x_dev = 0.0;
	double y_dev = 0.0; 
	double product = 0.0; 
	double sum = 0.0; 
	
	double x_mean = mean(x); 
	double y_mean = mean(y); 


	for (size_t i = 0; i < x.size(); i++) {
		x_dev = x[i] - x_mean; 
		y_dev = y[i] - y_mean; 
		product = x_dev * y_dev; 
		sum += product; 
	}


	if (x.size() == 1) { return 0.0; }

	return sum / (x.size() - 1); 
}

void LinearRegression::fit(const std::vector<double>& x, const std::vector<double>& y) {
	m_mean_x = mean(x); 
	m_covariance = covariance(x, y); 
	m_variance = variance(x); 

	// slope / intercept 
	m_slope = covariance(x, y) / variance(x); 
	m_intercept = mean(y) - (m_slope * mean(x)); 

}

std::vector<double> LinearRegression::predict(const std::vector<double>& x) {

	std::vector<double> y_pred; 

	for (double val : x) {
		
		double pred = (m_slope * val) + m_intercept; 
		y_pred.push_back(pred); 
	}

	return y_pred; 
}

double LinearRegression::MSE(const std::vector<double>& y, const std::vector<double>& y_pred) const {

	assert(y_pred.size() == y.size() && "Vector sizes do not match!");


	size_t sample_size = y_pred.size(); 

	double MSE = 0.0; 

	for (size_t i = 0; i < sample_size; i++) {

		MSE +=  pow(y[i] - y_pred[i], 2);
	}

	return MSE / sample_size; 
}



