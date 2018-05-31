#include "corners-data-wrapper.h"
#include "solution.h"
#include "rest-manager.h"

#include <iostream>
#include <boost/shared_ptr.hpp>

using namespace std;

void create_solution(const string& descriptor) {

	boost::shared_ptr<CornersDataWrapper> data_wrapper(new CornersDataWrapper());

	Solution solution(data_wrapper, descriptor);

	solution.evaluate();

	RestManager rest_manager;

	FitnessRecord record;
	record.descriptor = solution.get_descriptor().to_string();
	record.fitness = solution.get_fitness();
	record.n_weights = solution.get_weights();

	stringstream ss;
	ss << fixed << setprecision(2);

	vector<float> output_vector = solution.get_output_vector();

	for (unsigned int i = 0; i < output_vector.size() - 1; ++i) {
		ss << output_vector[i] << ";";
	}

	ss << *(output_vector.rbegin());

	record.training_input = ss.str();

	rest_manager.insert(record);

	cout << "Fitness: " << solution.get_fitness() << endl;
	cout << "Weights: " << solution.get_weights() << endl;

}

int main(int argc, char *argv[]) {

	if (argc < 2) {
		cout << "Usage: main <descriptor>" << endl;
		return EXIT_FAILURE;
	}

	srand(2);

	google::InitGoogleLogging("GA-DL");
	google::SetCommandLineOption("GLOG_minloglevel", "1");

	create_solution(argv[1]);

	return EXIT_SUCCESS;


}

