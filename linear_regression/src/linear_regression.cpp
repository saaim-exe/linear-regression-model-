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

	if (x.empty()) {
		throw std::invalid_argument("X vector is empty");
	}
	if (x.size() != y.size()) {
		throw std::invalid_argument("X and Y vector sizes do not match");
	}
	
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

	if (y_pred.size() != y_actual.size()) {
		throw std::invalid_argument("Prediction size must match target size");
	}


	size_t sample_size = y_pred.size(); 

	double MSE = 0.0; 

	for (size_t i = 0; i < sample_size; i++) {

		MSE +=  pow(y_actual[i] - y_pred[i], 2);
	}

	return MSE / sample_size; 
}

double SimpleLinearRegression::R_squared(const std::vector<double>& y_actual, const std::vector<double>& y_pred) const {

	if (y_pred.size() != y_actual.size()) {
		throw std::invalid_argument("Prediction size must match target size");
	}
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

	if (x.size() != y_actual.size() || y_actual.size() != y_pred.size()) {
		throw std::invalid_argument("X, target, and prediction sizes must match");
	}

	double weight_sum = 0.0; 
	double bias_sum = 0.0; 

	size_t sample_size = y_pred.size(); 

	for (size_t i = 0; i < sample_size; ++i) {

		double error = y_pred[i] - y_actual[i]; // 

		weight_sum += 2 * x[i] * (error);
		bias_sum += 2 * (error); 

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
		if (parsed_features[i].size() != n_features) {
			throw std::invalid_argument("Inconsistent feature count in parsed dataset");
		}

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

	if (test_size <= 0.0 || test_size >= 1.0) {
		throw std::invalid_argument("test_size must be between 0 and 1");
	}
	if (data.features.rows() != data.targets.size()) {
		throw std::invalid_argument("Feature rows must match target size");
	}
	if (data.features.rows() == 0) {
		throw std::invalid_argument("Cannot split an empty dataset");
	}

	std::random_device rd;
	std::default_random_engine rng(rd()); 

	size_t n_samples = static_cast<size_t>(data.features.rows()); 
	size_t n_features = static_cast<size_t>(data.features.cols()); 

	std::vector<size_t> indices(n_samples); 
	std::iota(indices.begin(), indices.end(), 0); 

	std::ranges::shuffle(indices, rng); 


	// 80 / 20 split 

	size_t test_count = static_cast<size_t>(std::round(n_samples * test_size)); 
	size_t train_count = n_samples - test_count; 

	MLR_subset s; 

	s.X_train.resize(train_count, n_features); 
	s.X_test.resize(test_count, n_features); 
	s.y_train.resize(train_count); 
	s.y_test.resize(test_count); 

	for (size_t i = 0; i < n_samples; ++i) {

		size_t idx = indices[i]; 

		if (i < test_count) {

			s.X_test.row(i) = data.features.row(idx); 
			s.y_test(i) = data.targets(idx); 
		}
		else {

			size_t train_row = i - test_count; 

			s.X_train.row(train_row) = data.features.row(idx);
			s.y_train(train_row) = data.targets(idx); 	
		}

	}
	return s; 
}



NormalizationStats MultipleLinearRegression::fit_normalizer(const MatrixXd& X) const {

	if (X.rows() == 0 || X.cols() == 0) {
		throw std::invalid_argument("Cannot normalize an empty feature matrix");
	}

	Index n_features = static_cast<size_t>(X.cols());

	VectorXd X_means(n_features); 
	VectorXd X_ranges(n_features); 
	
	X_means = X.colwise().mean().transpose(); 
	X_ranges = (X.colwise().maxCoeff() - X.colwise().minCoeff()).transpose(); 

	for (Index i = 0; i < n_features; ++i) { // AVOIDING DIVISON BY ZERO 
		if (X_ranges(i) == 0.0) {
			X_ranges(i) = 1.0; 
		}

	}

	NormalizationStats ns{
		.means = X_means,
		.ranges = X_ranges
	}; 

	return ns; 
}

MatrixXd MultipleLinearRegression::normalize(const MatrixXd& X, const NormalizationStats& stats) const {


	Index n_features = X.cols(); 

	if (stats.means.size() != n_features || stats.ranges.size() != n_features) {
		throw std::invalid_argument("Normalizer stats do not match feature count");
	}

	MatrixXd X_norm = X; 


	for (Index i = 0; i < n_features; ++i) {

		X_norm.col(i).array() -= stats.means(i); 
		X_norm.col(i).array() /= stats.ranges(i); 
	}

	return X_norm; 
}


parameters_mlr MultipleLinearRegression::fit(const MatrixXd& X_train, const VectorXd& y_train) {

	if (X_train.rows() != y_train.size()) {
		throw std::invalid_argument("X_train rows must match y_train size");
	}

	Index rows = X_train.rows(); 
	Index cols = X_train.cols(); 
		
	// add additional col for bias term 
	MatrixXd X_design(rows, cols + 1);
	X_design.col(0).setOnes();  // first col 

	X_design.rightCols(cols) = X_train; // features take up other cols 
	
	// closed form normal equation operations 

	auto X_t = X_design.transpose(); 
	auto gram = X_t * X_design; 
	auto projection = X_t * y_train; 

	VectorXd weights = gram.ldlt().solve(projection); 

	parameters_mlr p{
		.weights = weights.tail(cols),
		.bias = weights(0)
	};




	return p; 
}


VectorXd MultipleLinearRegression::predict(const MatrixXd& X, const parameters_mlr& p) const {

	if (X.cols() != p.weights.size()) {
		throw std::invalid_argument("Feature count must match weights size");
	}

	return (X * p.weights).array() + p.bias; 
}

double MultipleLinearRegression::MSE(const VectorXd& y_pred, const VectorXd& y_test) const {

	if (y_pred.size() != y_test.size()) {
		throw std::invalid_argument("Prediction size must match target size");
	}

	return (y_test - y_pred).array().square().mean();
}


double MultipleLinearRegression::R_squared(const VectorXd& y_pred, const VectorXd& y_test) const {

	if (y_pred.size() != y_test.size()) {
		throw std::invalid_argument("Prediction size must match target size");
	}

	double mean_y_test = y_test.mean(); 
	
	double SSR = (y_test - y_pred).squaredNorm(); 
	double SST = (y_test.array() - mean_y_test).pow(2).sum(); 

	if (SST == 0.0) {
		return 0.0; 
	}

	return (1 - (SSR / SST)); 
}

double MultipleLinearRegression::adjusted_R_squared(const MatrixXd& X, const VectorXd& y_pred, const VectorXd& y_test) const {

	if (X.rows() != y_test.size()){
		throw std::invalid_argument("X rows must match y_test size");
		
		std::cout << "X Rows: " << X.rows() << '\n';
		std::cout << "Y Test Rows: " << y_test.size(); 

	}

	Index n = X.rows(); 
	Index k = X.cols(); 


	double r_2 = R_squared(y_pred, y_test); 

	if ((n - k - 1) <= 0) {
		return 0.0; 
	}

	return 1.0 - ((1.0 - r_2) * (n - 1) / (n - k - 1));
}

parameters_mlr MultipleLinearRegression::gradient_descent(const MatrixXd& X_train, const VectorXd& y_train, const parameters_mlr& current_p, double learning_rate) const {

	if (X_train.rows() != y_train.size()) {
		throw std::invalid_argument("X_train rows must match y_train size");
	}
	if (X_train.cols() != current_p.weights.size()) {
		throw std::invalid_argument("Feature count must match weights size");
	}

	Index n_samples = X_train.rows();
	MatrixXd X_t = X_train.transpose();
	
	VectorXd weights = current_p.weights; 
	double bias = current_p.bias; 

	VectorXd y_pred = predict(X_train, current_p); 
	VectorXd error = y_pred - y_train; 

	VectorXd grad_weights = (2.0 / n_samples) * (X_t * error);
	double grad_bias = (2.0 / n_samples) * error.sum();

	weights -= learning_rate * grad_weights; 
	bias -= learning_rate * grad_bias;

	parameters_mlr p{
		.weights = weights,
		.bias = bias
	}; 

	return p; 
}

