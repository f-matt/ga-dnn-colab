#ifndef REGRESSOR_H_
#define REGRESSOR_H_

#include "descriptors.h"
#include "config.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <caffe/caffe.hpp>
#include <caffe/sgd_solvers.hpp>

using namespace caffe;
using namespace boost::filesystem;

template<class T>
class Regressor {

public:

	virtual void train() = 0;

	virtual float train_with_batch(const vector<T>& batch) = 0;

	virtual void regress(const T& pattern, vector<float> &pred, float &loss) = 0;

	virtual void regress(T& pattern) = 0;


	Regressor(const TopologyDescriptor &descriptor,
			int output_size,
			int batch_size,
			int max_epochs,
			int n_training_patterns,
			int n_test_patterns,
			int patience) :
		descriptor(descriptor),
		solver(nullptr),
		parameters(new NetParameter()),
		batch_size(batch_size),
		output_size(output_size),
		n_weights(0),
		test_loss(1e10),
		n_test_patterns(n_test_patterns),
		n_training_patterns(n_training_patterns),
		patience(patience),
		max_epochs(max_epochs) {

	}


	Regressor(const TopologyDescriptor &descriptor,
			int output_size,
			int batch_size,
			int max_epochs,
			int patience) :
		descriptor(descriptor),
		solver(nullptr),
		parameters(new NetParameter()),
		batch_size(batch_size),
		output_size(output_size),
		n_weights(0),
		test_loss(1e10),
		n_test_patterns(0),
		n_training_patterns(0),
		patience(patience),
		max_epochs(max_epochs) {

	}


	Regressor() : solver(nullptr),
			parameters(nullptr),
			batch_size(0),
			output_size(0),
			n_weights(0),
			test_loss(0),
			n_test_patterns(0),
			n_training_patterns(0),
			patience(0),
			max_epochs(0) {

	}


	virtual ~Regressor() {

	}


	void debug_weights() {

		const vector<boost::shared_ptr<Layer<float>>> layers = net->layers();
		boost::shared_ptr<Blob<float>> weights_blob = layers[1]->blobs()[0];

		size_t elements = 1;

		for (size_t i = 0; i < weights_blob->num_axes(); ++i)
			elements *= weights_blob->shape(i);

		const float* begin = weights_blob->cpu_data();
		const float* end = begin + elements;
		vector<float> weights = vector<float>(begin, end);

		cout << "First 3 training weights:" << endl;

		for (int i = 0; i < 3; ++i) {
			cout << "[" << i << "] -> " << weights[i] << endl;
		}

		const vector<boost::shared_ptr<Layer<float>>> test_layers = test_net->layers();
		boost::shared_ptr<Blob<float>> test_weights_blob = test_layers[1]->blobs()[0];

		elements = 1;

		for (size_t i = 0; i < test_weights_blob->num_axes(); ++i)
			elements *= test_weights_blob->shape(i);

		begin = test_weights_blob->cpu_data();
		end = begin + elements;
		vector<float> test_weights = vector<float>(begin, end);

		cout << "First 3 test weights:" << endl;

		for (int i = 0; i < 3; ++i) {
			cout << "[" << i << "] -> " << weights[i] << endl;
		}

	}


	float get_training_loss() {

		boost::shared_ptr<Blob<float>> loss = net->blob_by_name("loss");

		return loss->cpu_data()[0];
	}


	void snapshot() {

		// Clear snapshot directory (only keep the best snapshot)
		path p ("snapshots");

		directory_iterator end_itr;

		for (directory_iterator itr(p); itr != end_itr; ++itr) {
			if (is_regular_file(itr->path()))
				remove_all(itr->path());
		}

		solver->Snapshot();
	}


	int n_params() {
		return test_net->params().size();
	}


	long get_n_weights() {
		return n_weights;
	}


	float get_test_loss() {
		return test_loss;
	}


	void init() {

		// Create net from topology
		// Input layers
		add_input_layer();

		string previous_layer_name = "input";

		for (unsigned int i = 0; i < descriptor.get_conv_descriptors().size(); ++i) {
			string layer_name = "conv_" + to_string(i);
			previous_layer_name = add_conv_layer(descriptor.get_conv_descriptors()[i],
					layer_name,
					previous_layer_name);
		}

		for (unsigned int i = 0; i < descriptor.get_fc_descriptors().size(); ++i) {
			string layer_name = "fc_" + to_string(i);
			previous_layer_name = add_inner_product_layer(descriptor.get_fc_descriptors()[i],
					layer_name,
					previous_layer_name);
		}

		// Output layer
		FcDescriptor fcd(output_size, "L", 0);
		string output_layer_name = "output";
		previous_layer_name = add_inner_product_layer(fcd, output_layer_name, previous_layer_name);

		add_euclidean_loss_layer();

		SolverParameter solver_param;
		solver_param.set_base_lr(0.001);
		solver_param.set_lr_policy("fixed");
		solver_param.set_type("Adam");

		solver_param.set_random_seed(4096);
		solver_param.set_snapshot_prefix("snapshots/net");

		solver_param.set_solver_mode(SolverParameter_SolverMode_CPU);

		// Set iter to 0 to disable test during training
		solver_param.add_test_iter(0);
		solver_param.set_test_interval(100000);
		solver_param.set_test_initialization(false);

		solver_param.set_allocated_net_param(parameters.get());

		solver.reset(new caffe::AdamSolver<float>(solver_param));

		solver_param.release_net_param();

		net = solver->net();

		if (solver->test_nets().size() < 1) {
			cerr << "Test nets < 1" << endl;
			exit(EXIT_FAILURE);
		}

		test_net = solver->test_nets()[0];

		test_net->ShareTrainedLayersWith(net.get());

		update_weight_count();

	}


	void update_weight_count() {
		n_weights = 0;
		for (boost::shared_ptr<Layer<float>> layer : test_net->layers()) {
			for (boost::shared_ptr<Blob<float>> blob : layer->blobs()) {
				n_weights += blob->count();
			}
		}
	}

protected:

	virtual void add_input_layer() = 0;

	string add_conv_layer(const ConvDescriptor& cd, string& top, const string& bottom) {

		LayerParameter *layer_parameters = parameters->add_layer();
		layer_parameters->set_name(top);
		layer_parameters->set_type("Convolution");
		layer_parameters->add_top(top);
		layer_parameters->add_bottom(bottom);

		ParamSpec* conv_param_spec_1 = layer_parameters->add_param();
		conv_param_spec_1->set_lr_mult(1);
		conv_param_spec_1->set_decay_mult(1);

		ParamSpec* conv_param_spec_2 = layer_parameters->add_param();
		conv_param_spec_2->set_lr_mult(2);
		conv_param_spec_2->set_decay_mult(0);

		ConvolutionParameter *conv_parameters = new ConvolutionParameter();
		conv_parameters->set_num_output(cd.get_n_filters());
		conv_parameters->add_kernel_size(cd.get_kernel_size());
		conv_parameters->add_stride(cd.get_stride());

		FillerParameter *weight_filler = new FillerParameter();
		weight_filler->set_type("gaussian");
		weight_filler->set_std(0.01);

		FillerParameter *bias_filler = new FillerParameter();
		bias_filler->set_type("constant");
		bias_filler->set_value(0);

		conv_parameters->set_allocated_weight_filler(weight_filler);
		conv_parameters->set_allocated_bias_filler(bias_filler);

		layer_parameters->set_allocated_convolution_param(conv_parameters);

		// Activation function
		if (cd.get_activation_function() == "S")
			top = add_sigmoid(top);
		else if (cd.get_activation_function() == "T")
			top = add_tanh(top);
		else if (cd.get_activation_function() == "R")
			top = add_relu(top);

		// Pooling layer
		if (cd.get_pooling() > 0)
			return add_pooling_layer(cd, top);
		else
			return top;

	}


	string add_pooling_layer(const ConvDescriptor& cd, const string& conv_layer_name) {

		String layer_name = conv_layer_name + "_pool";

		LayerParameter *layer_parameters = parameters->add_layer();
		layer_parameters->set_name(layer_name);
		layer_parameters->set_type("Pooling");
		layer_parameters->add_top(layer_name);
		layer_parameters->add_bottom(conv_layer_name);

		PoolingParameter *pool_parameters = new PoolingParameter();
		pool_parameters->set_pool(PoolingParameter_PoolMethod_MAX);
		pool_parameters->set_kernel_size(cd.get_pooling());

		layer_parameters->set_allocated_pooling_param(pool_parameters);

		return layer_name;

	}


	string add_inner_product_layer(const FcDescriptor& descriptor, string& top, const string& bottom) {
		LayerParameter *fc1_parameters = parameters->add_layer();
		fc1_parameters->set_name(top);
		fc1_parameters->set_type("InnerProduct");
		fc1_parameters->add_top(top);
		fc1_parameters->add_bottom(bottom);

		ParamSpec* fc1_param_spec_1 = fc1_parameters->add_param();
		fc1_param_spec_1->set_lr_mult(1);
		fc1_param_spec_1->set_decay_mult(1);

		ParamSpec* fc1_param_spec_2 = fc1_parameters->add_param();
		fc1_param_spec_2->set_lr_mult(2);
		fc1_param_spec_2->set_decay_mult(0);

		InnerProductParameter *fc1_ip_parameters = new InnerProductParameter();
		fc1_ip_parameters->set_num_output(descriptor.get_n_neurons());

		FillerParameter *weight_filler = new FillerParameter();
		weight_filler->set_type("gaussian");
		weight_filler->set_std(0.01);

		FillerParameter *bias_filler = new FillerParameter();
		bias_filler->set_type("constant");
		bias_filler->set_value(0);

		fc1_ip_parameters->set_allocated_weight_filler(weight_filler);
		fc1_ip_parameters->set_allocated_bias_filler(bias_filler);

		fc1_parameters->set_allocated_inner_product_param(fc1_ip_parameters);

		// Activation function
		if (descriptor.get_activation_function() == "S")
			top = add_sigmoid(top);
		else if (descriptor.get_activation_function() == "T")
			top = add_tanh(top);
		else if (descriptor.get_activation_function() == "R")
			top = add_relu(top);

		if (descriptor.get_dropout() > 0) {
			return add_dropout_layer(descriptor, top);
		}

		return top;
	}


	string add_dropout_layer(const FcDescriptor& descriptor, const string& fc_layer_name) {

		string layer_name = fc_layer_name + "_dropout";

		LayerParameter *layer_parameters = parameters->add_layer();
		layer_parameters->set_name(layer_name);
		layer_parameters->set_type("Dropout");
		layer_parameters->add_top(layer_name);
		layer_parameters->add_bottom(fc_layer_name);

		DropoutParameter *dropout_parameters = new DropoutParameter();
		dropout_parameters->set_dropout_ratio(descriptor.get_dropout());

		layer_parameters->set_allocated_dropout_param(dropout_parameters);

		return layer_name;
	}


	string add_concat_layer(const string& top, const string& bottom1, const string& bottom2) {
		LayerParameter *concat_layer_parameters = parameters->add_layer();
		concat_layer_parameters->set_name(top);
		concat_layer_parameters->set_type("Concat");
		concat_layer_parameters->add_top(top);
		concat_layer_parameters->add_bottom(bottom1);
		concat_layer_parameters->add_bottom(bottom2);

		ConcatParameter *concat_parameter = new ConcatParameter();
		concat_parameter->set_axis(1);

		concat_layer_parameters->set_allocated_concat_param(concat_parameter);

		return top;
	}


	string add_relu(const string& previous_layer_name) {

		LayerParameter *relu_layer_parameters = parameters->add_layer();

		string layer_name = previous_layer_name + "_relu";

		relu_layer_parameters->set_name(layer_name);
		relu_layer_parameters->set_type("ReLU");
		relu_layer_parameters->add_top(layer_name);
		relu_layer_parameters->add_bottom(previous_layer_name);

		return layer_name;

	}


	string add_sigmoid(const string& previous_layer_name) {

		LayerParameter *relu_layer_parameters = parameters->add_layer();

		string layer_name = previous_layer_name + "_sigmoid";

		relu_layer_parameters->set_name(layer_name);
		relu_layer_parameters->set_type("Sigmoid");
		relu_layer_parameters->add_top(layer_name);
		relu_layer_parameters->add_bottom(previous_layer_name);

		return layer_name;

	}


	string add_tanh(const string& previous_layer_name) {
		LayerParameter *relu_layer_parameters = parameters->add_layer();

		string layer_name = previous_layer_name + "_tanh";

		relu_layer_parameters->set_name(layer_name);
		relu_layer_parameters->set_type("TanH");
		relu_layer_parameters->add_top(layer_name);
		relu_layer_parameters->add_bottom(previous_layer_name);

		return layer_name;

	}


	void add_euclidean_loss_layer() {
		LayerParameter *loss_parameters = parameters->add_layer();
		loss_parameters->set_name("loss");
		loss_parameters->set_type("EuclideanLoss");
		loss_parameters->add_top("loss");
		loss_parameters->add_bottom("output");
		loss_parameters->add_bottom("target");
	}



	void step() {
		assert(net->phase() == caffe::TRAIN);

		solver->Step(1);
	}

	TopologyDescriptor descriptor;

	boost::shared_ptr<Solver<float>> solver;

	boost::shared_ptr<NetParameter> parameters;

	int batch_size;

	int output_size;

	long n_weights;

	float test_loss;

	int n_test_patterns;

	int n_training_patterns;

	int patience;

	int max_epochs;

	boost::shared_ptr<Net<float>> net;

	boost::shared_ptr<Net<float>> test_net;

};

#endif
