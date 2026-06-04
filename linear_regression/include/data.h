#include <string>
#include <fstream>
#include <sstream>
#include <string_view>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

struct dataset {
	std::vector<std::vector<double>> features; 
	std::vector<double> targets; 
};


class DataLoader {

public: 
	 
	dataset parseCSV(const std::string& path, bool has_header = true); 
	std::pair<std::vector<double>, std::vector<double>> train_test_split(const dataset& dataset, double test_size = 0.2); 

		
private: 


};