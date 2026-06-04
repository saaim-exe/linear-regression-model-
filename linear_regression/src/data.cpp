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


	this->m_dataset = dataset; 
	return dataset; 
}

std::pair<std::vector<double>, std::vector<double>> DataLoader::train_test_split(const dataset& dataset, double test_size = 0.2) {

	auto rng = std::default_random_engine{}; 
	std::ranges::shuffle(dataset, rng); // ?? 




}
