#include "corners-data-wrapper.h"
#include "solution.h"

#include <iostream>
#include <boost/shared_ptr.hpp>

using namespace std;

void create_solution(const string& descriptor) {

	boost::shared_ptr<CornersDataWrapper> data_wrapper(new CornersDataWrapper());

	Solution solution(data_wrapper, descriptor);

	solution.evaluate();

	cout << "Fitness: " << solution.get_fitness() << endl;
	cout << "Weights: " << solution.get_weights() << endl;

}

int main(int argc, char *argv[]) {

	if (argc < 2) {
		cout << "Usage: main <descriptor>" << endl;
		return EXIT_FAILURE;
	}

	create_solution(argv[1]);

	return EXIT_SUCCESS;


}

