#include "linear_regression.h"


double SimpleLinearRegression::mean(const std::vector<double>& values) const {


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

double SimpleLinearRegression::variance(const std::vector<double>& x) const {

	return covariance(x, x); 
}

double SimpleLinearRegression::covariance(const std::vector<double>& x, const std::vector<double>& y) const {

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

parameters SimpleLinearRegression::fit(const std::vector<double>& x, const std::vector<double>& y) {
	m_mean_x = mean(x); 
	m_covariance = covariance(x, y); 
	m_variance = variance(x); 

	// weight / bias 
	m_weight = covariance(x, y) / variance(x); 
	m_bias = mean(y) - (m_weight * mean(x)); 

	parameters p{
		.weight = m_weight,
		.bias = m_bias
	}; 

	return p; 
}

std::vector<double> SimpleLinearRegression::predict(const std::vector<double>& x) {

	std::vector<double> y_pred; 

	for (double val : x) {
		
		double pred = (m_weight * val) + m_bias; 
		y_pred.push_back(pred); 
	}

	return y_pred; 
}

double SimpleLinearRegression::MSE(const std::vector<double>& y_actual, const std::vector<double>& y_pred) const {

	assert(y_pred.size() == y_actual.size() && "Vector sizes do not match!");


	size_t sample_size = y_pred.size(); 

	double MSE = 0.0; 

	for (size_t i = 0; i < sample_size; i++) {

		MSE +=  pow(y_actual[i] - y_pred[i], 2);
	}

	return MSE / sample_size; 
}

double SimpleLinearRegression::R_squared(const std::vector<double>& y_actual, const std::vector<double>& y_pred) const {

	assert(y_pred.size() == y_actual.size() && "Vector sizes do not match!");
	size_t sample_size = y_pred.size();


	auto mean_y_actual = mean(y_actual); 
	double SST = 0.0; 
	double SSR = 0.0; 
	
	for (size_t i = 0; i < sample_size; ++i) {

		SST += pow(y_actual[i] - mean_y_actual, 2);
		SSR += pow(y_actual[i] - y_pred[i], 2); 
	}


	return (1 - (SSR / SST)); 
}


parameters SimpleLinearRegression::gradient_descent(const std::vector<double>& x,const std::vector<double>& y_actual, const std::vector<double>& y_pred, const parameters& current_p, double learning_rate = 0.01) const {

	double weight_sum = 0.0; 
	double bias_sum = 0.0; 

	size_t sample_size = y_pred.size(); 

	for (size_t i = 0; i < sample_size; ++i) {

		double error = y_actual[i] - y_pred[i]; 

		weight_sum += -2 * x[i] * (error);
		bias_sum += -2 * (error); 

	}

	double weight_gradient = weight_sum / sample_size; 
	double bias_gradient = bias_sum / sample_size; 

	parameters updated_p{
		.weight = current_p.weight - (weight_gradient * learning_rate),
		.bias = current_p.bias - (bias_gradient * learning_rate)
	}; 


	return updated_p; 
}

dataset MultipleLinearRegression::loadData(const std::string& path, bool hasHeader = true) const {

	// load data function 



}