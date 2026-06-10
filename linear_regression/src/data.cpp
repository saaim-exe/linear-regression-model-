#include "data.h"



dataset DataLoader::parseCSV(const std::string& path, bool has_header = true) {

	dataset dataset; 

	std::ifstream file(path); 

	if (!file.is_open()) {

		std::cerr << "File cannot be opened: " << path << std::endl; 
	}

	std::string line; 

	if (has_header && std::getline(file,line)) {
		
	}

	while (std::getline(file, line)) {
		
		if (line.empty()) continue; 

		std::stringstream ss(line); 
		std::string value_str; 
		std::vector<double> row_features;

		while (std::getline(ss, value_str, ',')) {
			row_features.push_back(std::stod(value_str)); 
		}


		if (!row_features.empty()) {
			dataset.targets.push_back(row_features.back()); // target last column
			row_features.pop_back(); 

		}

		dataset.features.push_back(row_features); 

	}


	return dataset; 
}

subset DataLoader::train_test_split(const dataset& data, double test_size = 0.2) {

	assert(test_size > 0.0 && test_size < 1.0); 

	auto rng = std::default_random_engine{}; 
	size_t n = data.targets.size(); 

	std::vector<size_t> indices(n); 
	std::iota(indices.begin(), indices.end(), 0);

	std::ranges::shuffle(indices, rng); 
	
	// 80/20 split 

	size_t test_count = static_cast<size_t>(std::round(n * test_size));
	
	
	subset s; 

	s.X_train.reserve(n - test_count); 
	s.X_test.reserve(test_count); 

	s.Y_train.reserve(n - test_count); 
	s.Y_test.reserve(test_count); 
	

	for (size_t i = 0; i < n; ++i) {

		size_t idx = indices[i]; 

		if (i < test_count) {
			s.X_test.push_back(data.features[idx]); 
			s.Y_test.push_back(data.targets[idx]); 
		}
		else {
			s.X_train.push_back(data.features[idx]); 
			s.Y_train.push_back(data.targets[idx]); 
		}
	} 


	return s; 
}
