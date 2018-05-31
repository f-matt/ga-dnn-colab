#include "solution.h"

#include <iostream>
#include <sstream>

#include <boost/algorithm/string.hpp>

using namespace std;
using namespace boost;

const int N_OUTPUTS = 1;


Solution::Solution(boost::shared_ptr<CornersDataWrapper> &data_wrapper_ptr) :
				data_wrapper(data_wrapper_ptr),
				fitness(0),
				predicted(false) {

	init();

}


Solution::Solution(boost::shared_ptr<CornersDataWrapper> &data_wrapper_ptr, const string& descriptor) :
				data_wrapper(data_wrapper_ptr),
				fitness(0),
				predicted(false) {

	init(descriptor);

}


Solution::~Solution() {

}


// Initialize random layers
void Solution::init() {

	descriptor.randomize();

	while (!descriptor.is_valid(data_wrapper->get_width(), data_wrapper->get_height())) {
		descriptor.randomize();
	}

	// Predict if solution will be in the pareto front
	CornersRegressor regressor(descriptor,
			data_wrapper,
			OUTPUT_SIZE,
			BATCH_SIZE,
			MAX_EPOCHS,
			TRAIN_PATTERNS,
			TEST_PATTERNS,
			PATIENCE);

	n_weights = regressor.get_n_weights();

}


bool Solution::is_valid() {
	return descriptor.is_valid(data_wrapper->get_width(), data_wrapper->get_height());
}

// Initialize random layers
void Solution::init(const string& descriptor_str) {

	descriptor = TopologyDescriptor(descriptor_str);

	if (!descriptor.is_valid(data_wrapper->get_width(), data_wrapper->get_height())) {
		cerr << "Error: invalid descriptor " << descriptor_str << endl;
		exit(EXIT_FAILURE);
	}

	// Create regressor
	CornersRegressor regressor(descriptor,
			data_wrapper,
			OUTPUT_SIZE,
			BATCH_SIZE,
			MAX_EPOCHS,
			TRAIN_PATTERNS,
			TEST_PATTERNS,
			PATIENCE);

	n_weights = regressor.get_n_weights();

}


void Solution::evaluate() {

	// Create regressor
	CornersRegressor regressor(descriptor,
			data_wrapper,
			OUTPUT_SIZE,
			BATCH_SIZE,
			MAX_EPOCHS,
			TRAIN_PATTERNS,
			TEST_PATTERNS,
			PATIENCE);

	regressor.train();
	fitness = regressor.get_test_loss();
	n_weights = regressor.get_n_weights();

	predicted = false;
}


void Solution::print() {
	cout << "----------------" << endl;
	for (vector<float>::const_iterator it = descriptor.get_output_vector().begin();
			it != descriptor.get_output_vector().end();
			++it)
		cout << *it << endl;
	cout << "----------------" << endl;
}


float Solution::get_fitness() {
	return fitness;
}


void Solution::set_fitness(float fitness) {
	this->fitness = fitness;
}


bool Solution::operator< (const Solution &other) const {
	return fitness < other.fitness;
}


Solution& Solution::operator= (const Solution& other) {
	this->descriptor = other.descriptor;
	this->data_wrapper = other.data_wrapper;
	this->fitness = other.fitness;
	this->predicted = other.predicted;
	this->n_weights = other.n_weights;

	return *this;
}


void Solution::crossover_conv(Solution& other) {

	vector<ConvDescriptor> my_conv_descriptors = descriptor.get_conv_descriptors();
	vector<ConvDescriptor> other_conv_descriptors = other.descriptor.get_conv_descriptors();

	// Both descriptors are empty: return
	if (my_conv_descriptors.empty() && other_conv_descriptors.empty())
		return;

	// This descriptor is empty, other is not
	if (my_conv_descriptors.empty()) {
		int idx = rand() % other_conv_descriptors.size();
		my_conv_descriptors.push_back(other_conv_descriptors[idx]);
		my_conv_descriptors.erase(my_conv_descriptors.begin() + idx);
	} else if (other_conv_descriptors.empty()) {
		int idx = rand() % my_conv_descriptors.size();
		other_conv_descriptors.push_back(my_conv_descriptors[idx]);
		other_conv_descriptors.erase(other_conv_descriptors.begin() + idx);
	} else if (my_conv_descriptors.size() == 1) {
		// Swap crossover
		int idx = rand() % other_conv_descriptors.size();
		ConvDescriptor tmp = my_conv_descriptors[0];
		my_conv_descriptors[0] = other_conv_descriptors[idx];
		other_conv_descriptors[idx] = tmp;
	} else if (other_conv_descriptors.size() == 1) {
		int idx = rand() % my_conv_descriptors.size();
		ConvDescriptor tmp = other_conv_descriptors[0];
		other_conv_descriptors[0] = my_conv_descriptors[idx];
		my_conv_descriptors[idx] = tmp;
	} else {
		int cutpoint_1 = rand() % (my_conv_descriptors.size() - 1) + 1;
		int cutpoint_2 = rand() % (other_conv_descriptors.size() - 1) + 1;

		vector<ConvDescriptor> chrom1 = my_conv_descriptors;
		vector<ConvDescriptor> chrom2 = other_conv_descriptors;

		my_conv_descriptors.clear();
		other_conv_descriptors.clear();

		for (int i = 0; i < cutpoint_1; ++i)
			my_conv_descriptors.push_back(chrom1[i]);

		for (unsigned int i = cutpoint_2; i < chrom2.size(); ++i)
			my_conv_descriptors.push_back(chrom2[i]);

		for (int i = 0; i < cutpoint_2; ++i)
			other_conv_descriptors.push_back(chrom2[i]);

		for (unsigned int i = cutpoint_1; i < chrom1.size(); ++i)
			other_conv_descriptors.push_back(chrom1[i]);

	}

	change_conv_descriptors(my_conv_descriptors);
	other.change_conv_descriptors(other_conv_descriptors);

}


void Solution::crossover_fc(Solution& other) {

	vector<FcDescriptor> my_fc_descriptors = descriptor.get_fc_descriptors();
	vector<FcDescriptor> other_fc_descriptors = other.descriptor.get_fc_descriptors();

	// Both descriptors are empty: return
	if (my_fc_descriptors.empty() && other_fc_descriptors.empty())
		return;

	// This descriptor is empty, other is not
	if (my_fc_descriptors.empty()) {
		int idx = rand() % other_fc_descriptors.size();
		my_fc_descriptors.push_back(other_fc_descriptors[idx]);
		my_fc_descriptors.erase(my_fc_descriptors.begin() + idx);
	} else if (other_fc_descriptors.empty()) {
		int idx = rand() % my_fc_descriptors.size();
		other_fc_descriptors.push_back(my_fc_descriptors[idx]);
		other_fc_descriptors.erase(other_fc_descriptors.begin() + idx);
	} else if (my_fc_descriptors.size() == 1) {
		// Swap crossover
		int idx = rand() % other_fc_descriptors.size();
		FcDescriptor tmp = my_fc_descriptors[0];
		my_fc_descriptors[0] = other_fc_descriptors[idx];
		other_fc_descriptors[idx] = tmp;
	} else if (other_fc_descriptors.size() == 1) {
		// Swap crossover
		int idx = rand() % my_fc_descriptors.size();
		FcDescriptor tmp = other_fc_descriptors[0];
		other_fc_descriptors[0] = my_fc_descriptors[idx];
		my_fc_descriptors[idx] = tmp;
	} else {
		int cutpoint_1 = rand() % (my_fc_descriptors.size() - 1) + 1;
		int cutpoint_2 = rand() % (other_fc_descriptors.size() - 1) + 1;

		vector<FcDescriptor> chrom1 = my_fc_descriptors;
		vector<FcDescriptor> chrom2 = other_fc_descriptors;

		my_fc_descriptors.clear();
		other_fc_descriptors.clear();

		for (int i = 0; i < cutpoint_1; ++i)
			my_fc_descriptors.push_back(chrom1[i]);

		for (unsigned int i = cutpoint_2; i < chrom2.size(); ++i)
			my_fc_descriptors.push_back(chrom2[i]);

		for (int i = 0; i < cutpoint_2; ++i)
			other_fc_descriptors.push_back(chrom2[i]);

		for (unsigned int i = cutpoint_1; i < chrom1.size(); ++i)
			other_fc_descriptors.push_back(chrom1[i]);

	}

	change_fc_descriptors(my_fc_descriptors);
	other.change_fc_descriptors(other_fc_descriptors);

}


void Solution::mutate_conv() {

	vector<ConvDescriptor> conv_descriptors = descriptor.get_conv_descriptors();

	if (conv_descriptors.empty())
		return;

	// Layer which will suffer mutation
	int idx = rand() % conv_descriptors.size();

	conv_descriptors[idx].randomize();

	change_conv_descriptors(conv_descriptors);
}


void Solution::mutate_fc() {

	vector<FcDescriptor> fc_descriptors = descriptor.get_fc_descriptors();

	if (fc_descriptors.empty())
		return;

	// Layer which will suffer mutation
	int idx = rand() % fc_descriptors.size();

	fc_descriptors[idx].randomize();

	change_fc_descriptors(fc_descriptors);
}


long Solution::get_weights() {
	return n_weights;
}


vector<float> Solution::get_output_vector() {
	return descriptor.get_output_vector();
}


TopologyDescriptor Solution::get_descriptor() {
	return descriptor;
}


void Solution::change_conv_descriptors(vector<ConvDescriptor> descriptors) {
	descriptor.set_conv_descriptors(descriptors);

	while (!descriptor.is_valid(data_wrapper->get_width(), data_wrapper->get_height())) {
			cout << "Invalid descriptor: " << descriptor.to_string() << ". Generating new..." << endl;
			descriptor = TopologyDescriptor();
			descriptor.randomize();
	}

	CornersRegressor regressor(descriptor,
			data_wrapper,
			OUTPUT_SIZE,
			BATCH_SIZE,
			MAX_EPOCHS,
			TRAIN_PATTERNS,
			TEST_PATTERNS,
			PATIENCE);

	n_weights = regressor.get_n_weights();

}


void Solution::change_fc_descriptors(vector<FcDescriptor> descriptors) {
	descriptor.set_fc_descriptors(descriptors);

	while (!descriptor.is_valid(data_wrapper->get_width(), data_wrapper->get_height())) {
			cout << "Invalid descriptor: " << descriptor.to_string() << ". Generating new..." << endl;
			descriptor = TopologyDescriptor();
			descriptor.randomize();
	}

	CornersRegressor regressor(descriptor,
					data_wrapper,
					OUTPUT_SIZE,
					BATCH_SIZE,
					MAX_EPOCHS,
					TRAIN_PATTERNS,
					TEST_PATTERNS,
					PATIENCE);

	n_weights = regressor.get_n_weights();

}


bool Solution::is_predicted() {
	return predicted;
}


void Solution::set_predicted(bool m_predicted) {
	predicted = m_predicted;
}

