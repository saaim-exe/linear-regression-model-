#include <cmath>
#include <Eigen/Dense>


class LinearRegression {

public:
	
	LinearRegression() = default; // for now 

	double mean(const std::vector<double>& values) const; 
	double variance(const std::vector<double>& x) const;
	double covariance(const std::vector<double>& x, const std::vector<double>& y) const;
	void fit(const std::vector<double>& x, const std::vector<double>& y); 
	std::vector<double> predict(const std::vector<double>& x); 
	double MSE(const std::vector<double>& y, const std::vector<double>& y_pred) const; 


	// accessors 

	double getSlope() const {
		return m_slope; 
	}
	double getIntercept() const {
		return m_intercept; 
	}


private: 


	// slope = covariance of x,y / variance of x 
	double m_mean_x = 0.0;
	double m_covariance = 0.0; 
	double m_variance = 0.0; 
	double m_slope = 0.0; 
	double m_intercept = 0.0; 


};