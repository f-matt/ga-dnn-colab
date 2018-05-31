#ifndef TOPOLOGYGENERATOR_H_
#define TOPOLOGYGENERATOR_H_

#include <iostream>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <random>

#include <boost/algorithm/string.hpp>

using namespace std;

const vector<string> ACTIVATION_FUNCTION = {"L", "S", "T", "R"};

const int OUTPUT_VECTOR_SIZE = 39;
const int CONV_DESCRIPTOR_SIZE = 9;
const int FC_DESCRIPTOR_SIZE = 6;

const int N_CONV_SPEC_ELEMENTS = 5;
const int N_FC_SPEC_ELEMENTS = 3;

const int ADD_REMOVE_LAYER_PROB = 5;
const int MAX_CONV_LAYERS = 3;
const int MAX_FC_LAYERS = 2;

class ConvDescriptor {
public:
	ConvDescriptor();
	ConvDescriptor(int n_filters,
			int kernel_size,
			int stride,
			string activation_function,
			int pool_size,
			bool normalization);
	ConvDescriptor(string& spec);

	virtual ~ConvDescriptor();

	void randomize();
	vector<float> get_vector();
	bool empty();
	void add_to_stream(stringstream& ss);

	string to_string();

	int get_n_filters() const;
	int get_kernel_size() const;
	int get_stride() const;
	int get_pooling() const;
	string get_activation_function() const;

	static int get_n_filters_index(int filters);
	static int get_n_filters_by_index(unsigned int index);

	static int get_kernel_size_index(int kernel_size);
	static int get_kernel_size_by_index(unsigned int index);

	static int get_stride_index(int stride);
	static int get_stride_by_index(unsigned int index);

	static int get_pooling_index(int pooling);
	static int get_pooling_by_index(unsigned int index);

private:
	friend ostream& operator<<(ostream& os, const ConvDescriptor& cd);

	int n_filters;
	int kernel_size;
	int stride;

	string activation_function;

	int pool_size;

	bool normalization;

	static const vector<int> CONV_FILTERS;
	static const vector<int> CONV_KERNEL_SIZE;
	static const vector<int> CONV_STRIDE;
	static const vector<int> CONV_POOL_SIZE;
};


class FcDescriptor {
public:
	FcDescriptor();
	FcDescriptor(int n_neurons, string activation_function, float dropout);
	FcDescriptor(const string& spec);

	virtual ~FcDescriptor();

	void randomize();
	vector<float> get_vector();
	bool empty();
	void add_to_stream(stringstream& ss);

	string to_string();

	int get_n_neurons() const;
	float get_dropout() const;
	string get_activation_function() const;

	static int get_n_neurons_index(int n_neurons);
	static int get_n_neurons_from_index(unsigned int index);

	static int get_dropout_index(int dropout);
	static float get_dropout_from_index(unsigned int index);


private:
	friend ostream& operator<<(ostream& os, const FcDescriptor& fd);

	int n_neurons;
	string activation_function;
	float dropout;

	static const vector<int> FC_NEURONS;
	static const vector<float> FC_DROPOUT;
};


class TopologyDescriptor {
public:
	TopologyDescriptor();

	TopologyDescriptor(const string& descriptor);

	virtual ~TopologyDescriptor();

	void print_output_vector();

	void print();

	void randomize();

	void randomize_norm();

	string to_string() const;

	const vector<ConvDescriptor> get_conv_descriptors() const;

	void set_conv_descriptors(vector<ConvDescriptor>& conv_descriptors);

	const vector<FcDescriptor> get_fc_descriptors() const;

	void set_fc_descriptors(vector<FcDescriptor>& fc_descriptors);

	const vector<float> get_output_vector() const;

	bool is_valid(const int width, const int height) const;

private:
	void update_output_vector();

	ConvDescriptor sample_rand_conv_descriptor(const string& spec);
	FcDescriptor sample_rand_fc_descriptor(const string& spec);

	int sample_normal(float mean, float stdev);

	vector<ConvDescriptor> conv_descriptors;
	vector<FcDescriptor> fc_descriptors;

	vector<float> output_vector;
};

#endif
