#include "descriptors.h"

const vector<int> ConvDescriptor::CONV_FILTERS     = {2, 4, 8, 16, 32};
const vector<int> ConvDescriptor::CONV_KERNEL_SIZE = {3, 5, 7, 9, 11};
const vector<int> ConvDescriptor::CONV_STRIDE 	   = {1, 2, 3, 4};
const vector<int> ConvDescriptor::CONV_POOL_SIZE   = {2, 3, 4};


ConvDescriptor::ConvDescriptor() :
	n_filters(0),
	kernel_size(0),
	stride(0),
	activation_function(ACTIVATION_FUNCTION[0]),
	pool_size(0),
	normalization(false) {}


ConvDescriptor::ConvDescriptor(int n_filters,
		int kernel_size,
		int stride,
		string activation_function,
		int poll_size,
		bool normalization) :

		n_filters(n_filters),
		kernel_size(kernel_size),
		stride(stride),
		activation_function(activation_function),
		pool_size(poll_size),
		normalization(normalization) {}


ConvDescriptor::ConvDescriptor(string& spec) {
	vector<string> strs;

	boost::split(strs, spec, boost::is_any_of(";"));

	if (strs.size() != N_CONV_SPEC_ELEMENTS) {
		cerr << "Wrong number of elements in specification. Expected: " << N_CONV_SPEC_ELEMENTS << ". Found: " << strs.size() << endl;
		exit(EXIT_FAILURE);
	}

	// Layer type + activation function + normalization
	string layer = strs[0];

	if (layer[0] != 'C') {
		cerr << "Wrong specification (not a conv specification): " << layer << endl;
		exit(EXIT_FAILURE);
	}

	// Activation function
	activation_function = layer.substr(1, 1);

	// Normalization
	normalization = false;

	if (layer.back() == 'N')
		normalization = true;

	n_filters = stoi(strs[1]);
	kernel_size = stoi(strs[2]);
	stride = stoi(strs[3]);
	pool_size = stoi(strs[4]);

}


ConvDescriptor::~ConvDescriptor() {}

void ConvDescriptor::randomize() {

	n_filters = CONV_FILTERS[rand() % CONV_FILTERS.size()];
	kernel_size = CONV_KERNEL_SIZE[rand() % CONV_KERNEL_SIZE.size()];
	stride = CONV_STRIDE[rand() % CONV_STRIDE.size()];
	activation_function = ACTIVATION_FUNCTION[rand() % 4];

	pool_size = CONV_POOL_SIZE[rand() % CONV_POOL_SIZE.size()];
	normalization = rand() % 2 == 0 ? true : false;

}


vector<float> ConvDescriptor::get_vector() {

	vector<float> v(CONV_DESCRIPTOR_SIZE);

	if (n_filters == 0) {
		for (int i = 0; i < CONV_DESCRIPTOR_SIZE; ++i)
			v[i] = 0;
	} else {
		v[0] = n_filters;
		v[1] = kernel_size;
		v[2] = stride;

		for (int i = 3; i <= 6; ++i)
			v[i] = -1;

		if (activation_function == "L")
			v[3] = 1;
		else if (activation_function == "S")
			v[4] = 1;
		else if (activation_function == "T")
			v[5] = 1;
		else if (activation_function == "R")
			v[6] = 1;

		v[7] = pool_size;
		v[8] = normalization ? 1 : 0;
	}

	return v;

}


bool ConvDescriptor::empty() {
	if (n_filters > 0)
		return false;

	return true;
}


string ConvDescriptor::to_string() {
	stringstream ss;

	add_to_stream(ss);

	return ss.str();
}


void ConvDescriptor::add_to_stream(stringstream& ss) {
	if (!empty()) {
		if (ss.str().empty() || ss.str().back() == '|') {
			ss << (*this);
		} else {
			ss << "-" << (*this);
		}
	}
}


int ConvDescriptor::get_n_filters_by_index(unsigned int index) {
	if (index < 0)
		index = 0;
	else if (index > CONV_FILTERS.size() - 1)
		index = CONV_FILTERS.size() - 1;

	return CONV_FILTERS[index];
}


int ConvDescriptor::get_n_filters_index(int n_filters) {

	vector<int>::const_iterator it = find (CONV_FILTERS.begin(), CONV_FILTERS.end(), n_filters);

	if (it == CONV_FILTERS.end()) {
		cerr << "Error: number of filters not found - " << n_filters << endl;
		exit(EXIT_FAILURE);
	}

	return distance(CONV_FILTERS.begin(), it);

}


int ConvDescriptor::get_kernel_size_by_index(unsigned int index) {
	if (index < 0)
		index = 0;
	else if (index > CONV_KERNEL_SIZE.size() - 1)
		index = CONV_KERNEL_SIZE.size() - 1;

	return CONV_KERNEL_SIZE[index];
}


int ConvDescriptor::get_kernel_size_index(int n_filters) {

	vector<int>::const_iterator it = find (CONV_KERNEL_SIZE.begin(), CONV_KERNEL_SIZE.end(), n_filters);

	if (it == CONV_KERNEL_SIZE.end()) {
		cerr << "Error: number of filters not found - " << n_filters << endl;
		exit(EXIT_FAILURE);
	}

	return distance(CONV_KERNEL_SIZE.begin(), it);

}


int ConvDescriptor::get_stride_by_index(unsigned int index) {
	if (index < 0)
		index = 0;
	else if (index > CONV_STRIDE.size() - 1)
		index = CONV_STRIDE.size() - 1;

	return CONV_STRIDE[index];
}


int ConvDescriptor::get_stride_index(int stride) {

	vector<int>::const_iterator it = find (CONV_STRIDE.begin(), CONV_STRIDE.end(), stride);

	if (it == CONV_STRIDE.end()) {
		cerr << "Error: stride not found - " << stride << endl;
		exit(EXIT_FAILURE);
	}

	return distance(CONV_STRIDE.begin(), it);

}


int ConvDescriptor::get_pooling_by_index(unsigned int index) {
	if (index < 0)
		index = 0;
	else if (index > CONV_POOL_SIZE.size() - 1)
		index = CONV_POOL_SIZE.size() - 1;

	return CONV_POOL_SIZE[index];
}


int ConvDescriptor::get_pooling_index(int pool_size) {

	vector<int>::const_iterator it = find (CONV_POOL_SIZE.begin(), CONV_POOL_SIZE.end(), pool_size);

	if (it == CONV_POOL_SIZE.end()) {
		cerr << "Error: pool size not found - " << pool_size << endl;
		exit(EXIT_FAILURE);
	}

	return distance(CONV_POOL_SIZE.begin(), it);

}


int ConvDescriptor::get_n_filters() const {
	return n_filters;
}


int ConvDescriptor:: get_kernel_size() const {
	return kernel_size;
}


int ConvDescriptor::get_stride() const {
	return stride;
}


int ConvDescriptor::get_pooling() const {
	return pool_size;
}


string ConvDescriptor::get_activation_function() const {
	return activation_function;
}

ostream& operator<<(ostream& os, const ConvDescriptor& cd) {

	os << "C" << cd.activation_function;

	if (cd.normalization)
		os << "N";

	os << ";" << cd.n_filters << ";" << cd.kernel_size << ";" << cd.stride << ";" << cd.pool_size;

	return os;

}


const vector<int> FcDescriptor::FC_NEURONS = {8, 16, 32, 64};
const vector<float> FcDescriptor::FC_DROPOUT = {0.0, 0.2, 0.4, 0.6};


FcDescriptor::FcDescriptor() :
	n_neurons(0),
	activation_function(ACTIVATION_FUNCTION[0]),
	dropout(0) {}


FcDescriptor::FcDescriptor(int n_neurons, string activation_function, float dropout) :
	n_neurons(n_neurons),
	activation_function(activation_function),
	dropout(dropout) {}


FcDescriptor::FcDescriptor(const string& spec) {

	vector<string> strs;

	boost::split(strs, spec, boost::is_any_of(";"));

	if (strs.size() != N_FC_SPEC_ELEMENTS) {
		cerr << "Wrong number of elements in specification. Expected: " << N_FC_SPEC_ELEMENTS << ". Found: " << strs.size() << endl;
		exit(EXIT_FAILURE);
	}

	// Layer type + activation function
	string layer = strs[0];

	if (layer[0] != 'F') {
		cerr << "Wrong specification (not a FC specification): " << layer << endl;
		exit(EXIT_FAILURE);
	}

	activation_function = layer.substr(1,1);
	n_neurons = stoi(strs[1]);
	dropout = stof(strs[2]);

}


FcDescriptor::~FcDescriptor() {}


void FcDescriptor::randomize() {
	n_neurons = FC_NEURONS[rand() % FC_NEURONS.size()];

	activation_function = ACTIVATION_FUNCTION[rand() % 4];

	dropout = FC_DROPOUT[rand() % FC_DROPOUT.size()];
}


vector<float> FcDescriptor::get_vector() {

	vector<float> v(FC_DESCRIPTOR_SIZE);

	if (n_neurons == 0) {
		for (int i = 0; i < FC_DESCRIPTOR_SIZE; ++i)
			v[i] = 0;
	} else {
		v[0] = n_neurons;

		for (int i = 1; i <= 4; ++i)
			v[i] = -1;

		if (activation_function == "L")
			v[1] = 1;
		else if (activation_function == "S")
			v[2] = 1;
		else if (activation_function == "T")
			v[3] = 1;
		else if (activation_function == "R")
			v[4] = 1;

		v[5] = dropout;
	}

	return v;

}


bool FcDescriptor::empty() {
	if (n_neurons > 0)
		return false;

	return true;
}


string FcDescriptor::to_string() {
	stringstream ss;

	add_to_stream(ss);

	return ss.str();
}

void FcDescriptor::add_to_stream(stringstream& ss) {
	if (!empty()) {
		if (ss.str().empty() || ss.str().back() == '|') {
			ss << (*this);
		} else {
			ss << "-" << (*this);
		}
	}
}


int FcDescriptor::get_n_neurons_from_index(unsigned int index) {
	if (index < 0)
		index = 0;
	else if (index > FC_NEURONS.size() - 1)
		index = FC_NEURONS.size() - 1;

	return FC_NEURONS[index];
}


int FcDescriptor::get_n_neurons_index(int n_neurons) {

	vector<int>::const_iterator it = find (FC_NEURONS.begin(), FC_NEURONS.end(), n_neurons);

	if (it == FC_NEURONS.end()) {
		cerr << "Error: number of neurons not found - " << n_neurons << endl;
		exit(EXIT_FAILURE);
	}

	return distance(FC_NEURONS.begin(), it);

}


float FcDescriptor::get_dropout_from_index(unsigned int index) {
	if (index < 0)
		index = 0;
	else if (index > FC_DROPOUT.size() - 1)
		index = FC_DROPOUT.size() - 1;

	return FC_DROPOUT[index];
}


int FcDescriptor::get_dropout_index(int dropout) {

	vector<float>::const_iterator it = find (FC_DROPOUT.begin(), FC_DROPOUT.end(), dropout);

	if (it == FC_DROPOUT.end()) {
		cerr << "Error: dropout not found - " << dropout << endl;
		exit(EXIT_FAILURE);
	}

	return distance(FC_DROPOUT.begin(), it);

}


int FcDescriptor::get_n_neurons() const {
	return n_neurons;
}


float FcDescriptor::get_dropout() const {
	return dropout;
}


string FcDescriptor::get_activation_function() const {
	return activation_function;
}


ostream& operator<<(ostream& os, const FcDescriptor& fd) {

	os << "F" << fd.activation_function << ";" << fd.n_neurons << ";" << fd.dropout;

	return os;

}

TopologyDescriptor::TopologyDescriptor() {

	int n = rand() % 4;

	// Number of conv layers: 0 - 3
	for (int i = 0; i < n; ++i)
		conv_descriptors.push_back(ConvDescriptor());

	// Number of fc layers: 0 - 2
	n = rand() % 3;
	for (int i = 0; i < n; ++i)
		fc_descriptors.push_back(FcDescriptor());

	update_output_vector();

}


TopologyDescriptor::TopologyDescriptor(const string& descriptor) {

	// Main branch
	vector<string> strs;
	boost::split(strs, descriptor, boost::is_any_of("-"));

	if (strs.size() > 0) {

		while (!strs.empty() && (strs[0][0] == 'C')) {
			this->conv_descriptors.push_back(ConvDescriptor(strs[0]));
			strs.erase(strs.begin());
		}

		while (!strs.empty() && (strs[0][0] == 'F')) {
			this->fc_descriptors.push_back(FcDescriptor(strs[0]));
			strs.erase(strs.begin());
		}

	} else {
		cerr << "Error: invalid descriptor " << descriptor << endl;
		exit(EXIT_FAILURE);
	}

	update_output_vector();

}


TopologyDescriptor::~TopologyDescriptor() {}


/*
 * Randomize topology descriptor
 */
void TopologyDescriptor::randomize() {

	if (conv_descriptors.empty() && fc_descriptors.empty()) {

		int n = rand() % 4;

		// Number of conv layers: 0 - 3
		for (int i = 0; i < n; ++i) {
			ConvDescriptor cd;
			cd.randomize();
			conv_descriptors.push_back(cd);
		}

		// Number of fc layers: 0 - 2
		n = rand() % 3;
		for (int i = 0; i < n; ++i) {
			FcDescriptor fd;
			fd.randomize();
			fc_descriptors.push_back(fd);
		}
	} else {
		for (ConvDescriptor& cd : conv_descriptors)
			cd.randomize();

		for (FcDescriptor& fd : fc_descriptors)
			fd.randomize();
	}

	update_output_vector();

}

/*
 * Gaussian-based randomization of topology descriptor
 */
void TopologyDescriptor::randomize_norm() {

	// Left branch
	for (ConvDescriptor& cd : conv_descriptors)
		cd = sample_rand_conv_descriptor(cd.to_string());

	for (FcDescriptor& fcd : fc_descriptors)
		fcd = sample_rand_fc_descriptor(fcd.to_string());

	// Add/remove layers with 5% probability
	if (rand() % 100 < ADD_REMOVE_LAYER_PROB) {
		if ((rand() % 2 == 1) && (conv_descriptors.size() < MAX_CONV_LAYERS)) {
			ConvDescriptor cd;
			cd.randomize();
			conv_descriptors.push_back(cd);
		} else if (conv_descriptors.size() > 0) {
			conv_descriptors.erase(conv_descriptors.end());
		}
	}

	if (rand() % 100 < ADD_REMOVE_LAYER_PROB) {
		if ((rand() % 2 == 1) && (fc_descriptors.size() < MAX_FC_LAYERS)) {
			FcDescriptor fcd;
			fcd.randomize();
			fc_descriptors.push_back(fcd);
		} else if (fc_descriptors.size() > 0) {
			fc_descriptors.erase(fc_descriptors.end());
		}
	}

	update_output_vector();

}



void TopologyDescriptor::print_output_vector() {

	cout << "L-CONV-1" << endl;
	for (int i = 0; i < 9; ++i)
		cout << output_vector[i] << endl;

	cout << "L-CONV-2" << endl;
	for (int i = 9; i < 18; ++i)
		cout << output_vector[i] << endl;

	cout << "L-CONV-3" << endl;
	for (int i = 18; i < 27; ++i)
		cout << output_vector[i] << endl;

	cout << "L-FC-1" << endl;
	for (int i = 27; i < 33; ++i)
		cout << output_vector[i] << endl;

	cout << "L-FC-2" << endl;
	for (int i = 33; i < 39; ++i)
		cout << output_vector[i] << endl;

	cout << "R-CONV-1" << endl;
	for (int i = 39; i < 48; ++i)
		cout << output_vector[i] << endl;

	cout << "R-CONV-2" << endl;
	for (int i = 48; i < 57; ++i)
		cout << output_vector[i] << endl;

	cout << "R-CONV-3" << endl;
	for (int i = 57; i < 66; ++i)
		cout << output_vector[i] << endl;

	cout << "R-FC-1" << endl;
	for (int i = 66; i < 72; ++i)
		cout << output_vector[i] << endl;

	cout << "R-FC-2" << endl;
	for (int i = 72; i < 78; ++i)
		cout << output_vector[i] << endl;

	cout << "CONCAT-FC-1" << endl;
	for (int i = 78; i < 84; ++i)
		cout << output_vector[i] << endl;

	cout << "CONCAT-FC-2" << endl;
	for (int i = 84; i < 90; ++i)
		cout << output_vector[i] << endl;
}


void TopologyDescriptor::update_output_vector() {

	// Output vector (used as input for the ANN)
	output_vector = vector<float>(OUTPUT_VECTOR_SIZE);

	for (unsigned int i = 0; i < output_vector.size(); ++i)
		output_vector[i] = 0;

	vector<float>::iterator it = output_vector.begin();
	vector<float> descriptor;

	// Conv descriptors
	for (unsigned int i = 0; i < MAX_CONV_LAYERS; ++i) {
		if (conv_descriptors.size() >= (i + 1)) {
			descriptor = conv_descriptors[i].get_vector();
			copy_n(descriptor.begin(), descriptor.size(), it);
		}
		it += CONV_DESCRIPTOR_SIZE;
	}

	// FC descriptors
	for (unsigned int i = 0; i < MAX_FC_LAYERS; ++i) {
		if (fc_descriptors.size() >= (i + 1)) {
			descriptor = fc_descriptors[i].get_vector();
			copy_n(descriptor.begin(), descriptor.size(), it);
		}
		it += FC_DESCRIPTOR_SIZE;
	}

}


string TopologyDescriptor::to_string() const {

	stringstream ss;

	for (ConvDescriptor cd : conv_descriptors)
		cd.add_to_stream(ss);

	for (FcDescriptor fd : fc_descriptors)
		fd.add_to_stream(ss);

	return ss.str();

}


const vector<float> TopologyDescriptor::get_output_vector() const {

	return output_vector;

}


void TopologyDescriptor::print() {

	cout << to_string() << endl;

}



ConvDescriptor TopologyDescriptor::sample_rand_conv_descriptor(const string& spec) {

	vector<string> strs;

	boost::split(strs, spec, boost::is_any_of(";"));

	if (strs.size() != N_CONV_SPEC_ELEMENTS) {
		cerr << "Wrong number of elements in specification. Expected: " << N_CONV_SPEC_ELEMENTS << ". Found: " << strs.size() << endl;
		exit(EXIT_FAILURE);
	}

	// Layer type + activation function + normalization
	string layer = strs[0];

	if (layer[0] != 'C') {
		cerr << "Wrong specification (not a conv specification): " << layer << endl;
		exit(EXIT_FAILURE);
	}

	// Activation function
	vector<string>::const_iterator it = std::find(ACTIVATION_FUNCTION.begin(), ACTIVATION_FUNCTION.end(),
			layer.substr(1, 1));

	if (it == ACTIVATION_FUNCTION.end()) {
		cerr << "Error: activation function symbol not found - " << layer[1] << endl;
		exit(EXIT_FAILURE);
	}

	unsigned int new_activation_function_index = sample_normal(distance(ACTIVATION_FUNCTION.begin(), it), 1);

	if (new_activation_function_index < 0)
		new_activation_function_index = 0;
	else if (new_activation_function_index > ACTIVATION_FUNCTION.size() - 1)
		new_activation_function_index = ACTIVATION_FUNCTION.size() - 1;

	string new_activation_function = ACTIVATION_FUNCTION[new_activation_function_index];

	// Normalization
	bool normalization = false;

	if (layer.back() == 'N')
		normalization = true;

	// 80% to keep current, 20% to swap
	if (rand() % 100 < 20)
		normalization = !normalization;

	// Number of filters
	int n_filters_index = ConvDescriptor::get_n_filters_index(stoi(strs[1]));
	int new_n_filters_index = sample_normal(n_filters_index, 1);
	int new_n_filters = ConvDescriptor::get_n_filters_by_index(new_n_filters_index);

	// Kernel size
	int k_size_index = ConvDescriptor::get_kernel_size_index(stof(strs[2]));
	int new_kernel_size_index = sample_normal(k_size_index, 1);
	int new_kernel_size = ConvDescriptor::get_kernel_size_by_index(new_kernel_size_index);

	// Stride
	int stride_index = ConvDescriptor::get_stride_index(stof(strs[3]));
	int new_stride_index = sample_normal(stride_index, 1);
	int new_stride = ConvDescriptor::get_stride_by_index(new_stride_index);

	// Pooling
	int pooling_index = ConvDescriptor::get_pooling_index(stof(strs[4]));
	int new_pooling_index = sample_normal(pooling_index, 1);
	int new_pooling = ConvDescriptor::get_pooling_by_index(new_pooling_index);

	ConvDescriptor cd(new_n_filters, new_kernel_size, new_stride, new_activation_function, new_pooling, normalization);

	return cd;

}


FcDescriptor TopologyDescriptor::sample_rand_fc_descriptor(const string& spec) {

	vector<string> strs;

	boost::split(strs, spec, boost::is_any_of(";"));

	if (strs.size() != N_FC_SPEC_ELEMENTS) {
		cerr << "Wrong number of elements in specification. Expected: " << N_FC_SPEC_ELEMENTS << ". Found: " << strs.size() << endl;
		exit(EXIT_FAILURE);
	}

	// Layer type + activation function
	string layer = strs[0];

	if (layer[0] != 'F') {
		cerr << "Wrong specification (not a FC specification): " << layer << endl;
		exit(EXIT_FAILURE);
	}

	// Activation function
	vector<string>::const_iterator it = std::find(ACTIVATION_FUNCTION.begin(), ACTIVATION_FUNCTION.end(),
			layer.substr(1,1));

	if (it == ACTIVATION_FUNCTION.end()) {
		cerr << "Error: activation function symbol not found - " << layer[1] << endl;
		exit(EXIT_FAILURE);
	}

	unsigned int new_activation_function_index = sample_normal(distance(ACTIVATION_FUNCTION.begin(), it), 1);

	if (new_activation_function_index < 0)
		new_activation_function_index = 0;
	else if (new_activation_function_index > ACTIVATION_FUNCTION.size() - 1)
		new_activation_function_index = ACTIVATION_FUNCTION.size() - 1;

	string new_activation_function = ACTIVATION_FUNCTION[new_activation_function_index];

	// Number of neurons
	int n_neurons_index = FcDescriptor::get_n_neurons_index(stof(strs[1]));
	int new_n_neurons_index = sample_normal(n_neurons_index, 1);
	int new_n_neurons = FcDescriptor::get_n_neurons_from_index(new_n_neurons_index);

	// Dropout
	int dropout_index = FcDescriptor::get_dropout_index(stof(strs[2]));
	int new_dropout_index = sample_normal(dropout_index, 1);
	float new_dropout = FcDescriptor::get_dropout_from_index(new_dropout_index);

	FcDescriptor fcd(new_n_neurons, new_activation_function, new_dropout);

	return fcd;

}


int TopologyDescriptor::sample_normal(float mean, float stdev) {
	default_random_engine generator;
	normal_distribution<float> distribution(mean, stdev);

	return static_cast<int>(round(distribution(generator)));
}


const vector<ConvDescriptor> TopologyDescriptor::get_conv_descriptors() const {
	return conv_descriptors;
}


const vector<FcDescriptor> TopologyDescriptor::get_fc_descriptors() const {
	return fc_descriptors;
}


void TopologyDescriptor::set_conv_descriptors(vector<ConvDescriptor>& conv_descriptors) {
	this->conv_descriptors = conv_descriptors;
	update_output_vector();
}


void TopologyDescriptor::set_fc_descriptors(vector<FcDescriptor>& fc_descriptors) {
	this->fc_descriptors = fc_descriptors;
	update_output_vector();
}


bool TopologyDescriptor::is_valid(const int width, const int height) const {

	// Prevent empty solution
	if (conv_descriptors.empty() && fc_descriptors.empty())
		return false;

	int dim = std::min(width, height);

	for (ConvDescriptor cd : conv_descriptors) {
		dim = (dim - cd.get_kernel_size()) / cd.get_stride() + 1;
		dim /= cd.get_pooling();

		if (dim < 1) {
			return false;
		}

	}

	return true;

}




