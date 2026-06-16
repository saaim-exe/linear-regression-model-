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

parameters_slr SimpleLinearRegression::fit(const std::vector<double>& x, const std::vector<double>& y) {
	m_mean_x = mean(x); 
	m_covariance = covariance(x, y); 
	m_variance = variance(x); 

	// weight / bias 
	m_weight = covariance(x, y) / variance(x); 
	m_bias = mean(y) - (m_weight * mean(x)); 

	parameters_slr p{
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


parameters_slr SimpleLinearRegression::gradient_descent(const std::vector<double>& x,const std::vector<double>& y_actual, const std::vector<double>& y_pred, const parameters_slr& current_p, double learning_rate) const {

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

	parameters_slr updated_p{
		.weight = current_p.weight - (weight_gradient * learning_rate),
		.bias = current_p.bias - (bias_gradient * learning_rate)
	}; 


	return updated_p; 
}

MLR_dataset MultipleLinearRegression::loadData(const dataset& data) {

	// loop through features / targets and convert into eigen matrix (for feature) and column vector (for target) 

	auto parsed_features = data.features; 
	auto parsed_targets = data.targets; 

	if (parsed_features.empty()) {
		std::cerr << "0 entries in CSV" << std::endl; 
		return { MatrixXd{0, 0}, VectorXd{0} }; 
	}
	assert(parsed_targets.size() == parsed_features.size() && "Vector sizes do not match!"); 

	
	size_t n_samples = parsed_features.size();
	size_t n_features = parsed_features[0].size(); 

	MatrixXd features(n_samples, n_features); 
	VectorXd targets(n_samples); 

	for (size_t i = 0; i < n_samples; ++i) {

		targets(i) = parsed_targets[i]; 

		for (size_t j = 0; j < n_features; ++j) {

			features(i, j) = parsed_features[i][j]; 

		}
	}


	MLR_dataset mlr_data{
		.features = features,
		.targets = targets
	}; 


	return mlr_data; 
}

MLR_subset MultipleLinearRegression::train_test_split(const MLR_dataset& data, double test_size) {

	assert(test_size > 0.0 && test_size < 1.0); 
	
	auto rng = std::default_random_engine{}; 

	size_t n_samples = static_cast<size_t>(data.features.rows()); 
	size_t n_features = static_cast<size_t>(data.features.cols()); 
	size_t n_targets = static_cast<size_t>(data.targets.cols()); 

	std::vector<size_t> indices(n_samples); 
	std::iota(indices.begin(), indices.end(), 0); 

	std::ranges::shuffle(indices, rng); 


	// 80 / 20 split 

	size_t test_count = static_cast<size_t>(std::round(n_samples * test_size)); 
	size_t train_count = n_samples - test_count; 

	MLR_subset s; 

	s.X_train.resize(train_count, n_features); 
	s.X_test.resize(test_count, n_features); 
	s.y_train.resize(train_count, n_targets); 
	s.y_test.resize(test_count, n_targets); 

	for (size_t i = 0; i < n_samples; ++i) {

		size_t idx = indices[i]; 

		if (i < test_count) {

			s.X_test.row(i) = data.features.row(idx); 
			s.y_test.row(i) = data.targets.row(idx); 
		}
		else {

			size_t train_row = i - test_count; 

			s.X_train.row(train_row) = data.features.row(idx);
			s.y_train.row(train_row) = data.targets.row(idx); 	
		}

	}
	return s; 
}

MLR_dataset MultipleLinearRegression::normalize(MLR_dataset& data) const{ 


	MatrixXd features_t = data.features.transpose(); 
	auto n_features_t = static_cast<size_t>(features_t.rows()); 
	
	for (size_t i = 0; i < n_features_t; ++i) {
		auto fmean = features_t.row(i).mean(); 
		auto fmax = features_t.row(i).maxCoeff(); 
		auto fmin = features_t.row(i).minCoeff(); 
		double frange = fmax - fmin; 

		// AVOID DIVISION BY ZERO (if feature column has identical value) 
		if (frange == 0.0) {
			frange = 1.0; 
		}

		features_t.row(i).array() -= fmean;
		features_t.row(i).array() /= frange;
	}
	
	MLR_dataset normalized_data{
		.features = features_t.transpose(),
		.targets = data.targets
	}; 


	return normalized_data; 
}

parameters_mlr MultipleLinearRegression::fit(MLR_dataset& data) {

	auto norm_data = normalize(data); 

	auto norm_features = norm_data.features;
	auto norm_target = norm_data.targets;

	long rows = norm_data.features.rows(); 
	long cols = norm_data.features.cols(); 

	// add additional col for bias term 
	MatrixXd X(rows, cols + 1); 
	X.col(0).setOnes();  // first col 
	
	X.rightCols(cols) = norm_features; // features take up other cols 

	// closed form normal equation operations 
	auto X_t = X.transpose(); 
	auto gram = X_t * X; 
	auto projection = X_t * norm_target; 

	VectorXd weights = gram.ldlt().solve(projection); 

	parameters_mlr p{
		.weights = weights.tail(cols),
		.bias = weights(0)
	}; 

	return p; 
}

VectorXd MultipleLinearRegression::predict(const MatrixXd& X_test, const parameters_mlr& p) {

	return (X_test * p.weights).array() + p.bias; 
}

double MultipleLinearRegression::MSE(const VectorXd& y_pred, const VectorXd& y_test) const {

	return (y_test - y_pred).array().square().mean();
}


double MultipleLinearRegression::R_squared(const VectorXd& y_pred, const VectorXd& y_test) const {

	double mean_y_test = y_test.mean(); 
	
	double SSR = (y_test - y_pred).squaredNorm(); 
	double SST = (y_test.array() - mean_y_test).pow(2).sum(); 

	if (SST == 0.0) {
		return 0.0; 
	}

	return (1 - (SSR / SST)); 
}

double MultipleLinearRegression::adjusted_R_squared(const MLR_dataset& data, const VectorXd& y_pred, const VectorXd& y_test) const {

	Index n = static_cast<size_t>(data.features.rows()); 
	Index k = static_cast<size_t>(data.features.cols()); 

	double r_2 = R_squared(y_pred, y_test); 

	if ((n - k - 1) <= 0) {
		return 0.0; 
	}

	return 1.0 - ((1.0 - r_2) * static_cast<double>(n - 1) / static_cast<double>(n - k - 1));
}
