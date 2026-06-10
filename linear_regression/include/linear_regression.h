#include <cmath>
#include <vector>
#include <numeric>
#include <Eigen/Dense>
#include "data.h"


struct parameters {
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
	parameters fit(const std::vector<double>& x, const std::vector<double>& y); 
	std::vector<double> predict(const std::vector<double>& x); 
	double MSE(const std::vector<double>& y_actual, const std::vector<double>& y_pred) const;
	double R_squared(const std::vector<double>& y_actual, const std::vector<double>& y_pred) const; 
	parameters gradient_descent(const std::vector<double>& x,const std::vector<double>& y_actual,const std::vector<double>& y_pred, const parameters& current_p,  double learning_rate = 0.01) const;


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

class MultipleLinearRegression {

public: 

	dataset loadData(const std::string& path, bool hasHeader = true) const;





private: 

	std::vector<double> m_weights; 
	double m_bias; 



};


