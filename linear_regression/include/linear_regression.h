#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <Eigen/Dense>
#include "data.h"

using namespace Eigen; 

struct parameters_slr {
	double weight; 
	double bias; 
};

class SimpleLinearRegression {

public:
	
	SimpleLinearRegression() = default; // for now 

	// x - independent var vector // y - dependent var vector 

	double mean(const std::vector<double>& values) const; 
	double variance(const std::vector<double>& x) const;
	double covariance(const std::vector<double>& x, const std::vector<double>& y) const;
	parameters_slr fit(const std::vector<double>& x, const std::vector<double>& y); // closed form normal equation 
	std::vector<double> predict(const std::vector<double>& x); 
	double MSE(const std::vector<double>& y_actual, const std::vector<double>& y_pred) const;
	double R_squared(const std::vector<double>& y_actual, const std::vector<double>& y_pred) const; 
	parameters_slr gradient_descent(const std::vector<double>& x,const std::vector<double>& y_actual,const std::vector<double>& y_pred, const parameters_slr& current_p,  double learning_rate = 0.01) const;


	// accessors 

	double getSlope() const {
		return m_weight; 
	}
	double getIntercept() const {
		return m_bias; 
	}


private: 


	// slope = covariance of x,y / variance of x 
	double m_mean_x; 
	double m_covariance;
	double m_variance;
	double m_weight;
	double m_bias; 


}; 


struct parameters_mlr {

	VectorXd weights; 
	double bias; 
};

struct MLR_dataset {
	MatrixXd features;
	VectorXd targets;
};

struct MLR_subset {
	MatrixXd X_train; 
	MatrixXd X_test; 
	VectorXd y_train;
	VectorXd y_test;
};

struct NormalizationStats {
	VectorXd means; 
	VectorXd ranges; 
};



class MultipleLinearRegression {

public: 

	MLR_dataset loadData(const dataset& data);
	MLR_subset train_test_split(const MLR_dataset& data, double test_size = 0.2);

	NormalizationStats fit_normalizer(const MatrixXd& X) const; 
	MatrixXd normalize(const MatrixXd& X, const NormalizationStats& stats) const; 

	parameters_mlr fit(const MatrixXd& X_train, const VectorXd& y_train); // closed form normal equation



	VectorXd predict(const MatrixXd& X, const parameters_mlr& p) const; 
	double MSE(const VectorXd& y_pred, const VectorXd& y_test) const;
	double R_squared(const VectorXd& y_pred, const VectorXd& y_test) const; 
	double adjusted_R_squared(const MatrixXd& X_train, const VectorXd& y_pred, const VectorXd& y_test) const;
	// gradient descent 
	parameters_mlr gradient_descent(const MatrixXd& X_train, const VectorXd& y_train, const parameters_mlr& current_p, double learning_rate = 0.01) const;

private: 

	VectorXd m_weights; 
	double m_bias; 
	

};


