#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <string_view>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <assert.h>
#include <iostream>

// THIS IS FOR MLR 

struct dataset {
	std::vector<std::vector<double>> features; 
	std::vector<double> targets; 
};

struct subset {
	std::vector<std::vector<double>> X_train;
	std::vector<std::vector<double>> X_test; 
	std::vector<double> Y_train; 
	std::vector<double> Y_test; 
};


class DataLoader {

public: 
	 
	dataset parseCSV(const std::string& path, bool has_header = true); 
//	subset train_test_split(const dataset & data, double test_size = 0.2);

		
private: 


};