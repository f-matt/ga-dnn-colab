#include "corners-regressor.h"
#include "corners-data-wrapper.h"

#include <caffe/blob.hpp>
#include <caffe/net.hpp>
#include <caffe/proto/caffe.pb.h>
#include <opencv2/core/core.hpp>
#include <stddef.h>
#include <cassert>
#include <string>
#include <vector>


CornersRegressor::CornersRegressor(const TopologyDescriptor &descriptor,
		boost::shared_ptr<CornersDataWrapper> &data_wrapper,
		int output_size,
		int batch_size,
		int max_epochs,
		int n_training_patterns,
		int n_test_patterns,
		int patience) :
		Regressor(descriptor,
				output_size,
				batch_size,
				max_epochs,
				n_training_patterns,
				n_test_patterns,
				patience),
		data_wrapper(data_wrapper) {

	Regressor::init();

}


CornersRegressor::CornersRegressor(boost::shared_ptr<CornersDataWrapper> &data_wrapper,
		int output_size,
		int batch_size,
		int max_epochs,
		int n_training_patterns,
		int n_test_patterns,
		int patience) :
		Regressor(descriptor,
				output_size,
				batch_size,
				max_epochs,
				n_training_patterns,
				n_test_patterns,
				patience),
		data_wrapper(data_wrapper) {

}


CornersRegressor::CornersRegressor() : Regressor() {

}


CornersRegressor::~CornersRegressor() {

}


void CornersRegressor::add_input_layer() {
	LayerParameter *input_parameters = parameters->add_layer();
    input_parameters->set_name("input");
    input_parameters->set_type("Input");
    input_parameters->add_top("input");
    input_parameters->add_top("target");

    InputParameter *input_param = new InputParameter();

    BlobShape *image_shape = input_param->add_shape();
    image_shape->add_dim(1);
    image_shape->add_dim(data_wrapper->get_channels());
    image_shape->add_dim(data_wrapper->get_height());
    image_shape->add_dim(data_wrapper->get_width());

    BlobShape *output_shape = input_param->add_shape();
    output_shape->add_dim(1);
    output_shape->add_dim(OUTPUT_SIZE);
    output_shape->add_dim(1);
    output_shape->add_dim(1);

	input_parameters->set_allocated_input_param(input_param);

}


void CornersRegressor::train() {

	char progress[] = {'|', '/', '-', '\\'};

	// Load training and test patterns
	pair<vector<Pattern>, vector<Pattern>> train_test_data = data_wrapper->load_train_test_data(TRAIN_PATTERNS, TEST_PATTERNS);

	vector<Pattern> train_vector = train_test_data.first;

	vector<Pattern> test_vector = train_test_data.second;

	if (train_vector.size() % BATCH_SIZE != 0) {
		cerr << "The number of training patterns must be a multiple of the batch size" << endl;
		exit(-1);
	}

	int batches_per_epoch = train_vector.size() / BATCH_SIZE;

	vector <float> v_test_loss;
	float best_test_loss = 1e10;
	float best_train_loss = 1e10;
	int epochs_without_improvement = 0;

	for (size_t i = 0; i < MAX_EPOCHS; ++i) {

		float epoch_training_loss = 0;

		float avg_batch_processing_delay = -1.0;

		double elapsed;

		boost::posix_time::ptime start, stop;
		boost::posix_time::time_duration td;

		for (int j = 0; j < batches_per_epoch; ++j) {

			cout << boost::posix_time::to_simple_string(boost::posix_time::second_clock::local_time()) <<
					" Epoch: " << i + 1 << " " << progress[j % 4] <<
					" Avg delay: " << avg_batch_processing_delay << flush;

			vector<Pattern>::const_iterator first = train_vector.begin() + j * BATCH_SIZE;
			vector<Pattern>::const_iterator last = train_vector.begin() + (j+1) * BATCH_SIZE;

			vector<Pattern> batch(first, last);

			start = boost::posix_time::microsec_clock::local_time();
			epoch_training_loss += train_with_batch(batch);
			stop = boost::posix_time::microsec_clock::local_time();

			td = stop - start;

			elapsed = td.total_milliseconds();

			if (avg_batch_processing_delay < 0) {
				avg_batch_processing_delay = elapsed;
			} else {
				avg_batch_processing_delay = (avg_batch_processing_delay + elapsed) / 2;
			}

			cout << '\r';
		}

		epoch_training_loss /= batches_per_epoch;

		float epoch_test_loss = 0;

		for (size_t j = 0; j < test_vector.size(); ++j) {
			vector<float> pred(8);
			vector<float> truth = test_vector[j].coords;

			float loss;

			regress(test_vector[j], pred, loss);

			epoch_test_loss += loss;

		}

		epoch_test_loss /= test_vector.size();

		v_test_loss.push_back(epoch_test_loss);

		//cout << "[Epoch " << fixed << setw(4) << (i+1) << "] Train loss: " << setprecision(6) << epoch_training_loss <<
		//		" Test loss: " << epoch_test_loss << endl;

		if (epoch_training_loss < best_train_loss) {
			best_train_loss = epoch_training_loss;
		}

		if (epoch_test_loss < best_test_loss) {
			best_test_loss = epoch_test_loss;
			epochs_without_improvement = 0;
		} else {
			epochs_without_improvement++;
			if (epochs_without_improvement == PATIENCE) {
				break;
			}
		}
	}

	cout << '\r' << flush;

	// Update regressor best test loss
	test_loss = best_test_loss;

}


float CornersRegressor::train_with_batch(const vector<Pattern>& batch) {

	assert(net->phase() == caffe::TRAIN);

	const size_t num_images = batch.size();

	// Set network inputs to the appropriate size and number
	// First image
	Blob<float>* input_image_layer = net->input_blobs()[0];
	input_image_layer->Reshape(num_images, data_wrapper->get_channels(), data_wrapper->get_height(), data_wrapper->get_width());

	// To backprop, we need to input the ground-truth bounding boxes
	// Reshape the bounding boxes
	Blob<float>* input_gt = net->input_blobs()[1];
	input_gt->Reshape(num_images, OUTPUT_SIZE, 1, 1);

	// Forward reshape
	net->Reshape();

	// Get a pointer to the corners memory
	float* input_gt_data = input_gt->mutable_cpu_data();

	int input_gt_data_counter = 0;

	for (size_t i = 0; i < batch.size(); ++i) {
		const Pattern& patt = batch[i];

		// Set the corners data do the ground-truth corners
		for (unsigned int j = 0; j < patt.coords.size(); ++j)
			input_gt_data[input_gt_data_counter++] = patt.coords[j];

	}

	vector<vector<Mat>> image_channels;
	image_channels.resize(num_images);

	float* image_data = input_image_layer->mutable_cpu_data();

	for (size_t n = 0; n < num_images; ++n) {
		// First image
		Mat image_channel(data_wrapper->get_height(), data_wrapper->get_width(), CV_32FC1, image_data);
		image_channels[n].push_back(image_channel);
		image_data += data_wrapper->get_width() * data_wrapper->get_height();
		split(batch[n].image, image_channels[n]);
	}

	step();

	boost::shared_ptr<Blob<float>> loss = net->blob_by_name("loss");

	return loss->cpu_data()[0];

}


void CornersRegressor::regress(const Pattern& pattern, vector<float> &pred, float &loss) {

	assert(test_net->phase() == caffe::TEST);

	Blob<float>* input_image_layer = test_net->input_blobs()[0];
	input_image_layer->Reshape(1, data_wrapper->get_channels(), data_wrapper->get_height(), data_wrapper->get_width());

	Blob<float>* input_coords_delta_layer = test_net->input_blobs()[1];
	input_coords_delta_layer->Reshape(1, OUTPUT_SIZE, 1, 1);

	// Forward dimension change
	test_net->Reshape();

	// Wrap input layer to Mat's
	vector<Mat> image_channels;

	float* img_data = input_image_layer->mutable_cpu_data();

	for (int i = 0; i < data_wrapper->get_channels(); ++i) {
		Mat image_channel(data_wrapper->get_height(), data_wrapper->get_width(), CV_32FC1, img_data);
		image_channels.push_back(image_channel);
		img_data += data_wrapper->get_width() * data_wrapper->get_height();
	}

	split(pattern.image, image_channels);

	// Fill ground truth values
	float* input_coords_delta_data = input_coords_delta_layer->mutable_cpu_data();

	for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
		input_coords_delta_data[i] = pattern.coords[i];
	}

	test_net->Forward(&loss);

	boost::shared_ptr<Blob<float>> output_layer = test_net->blob_by_name("output");

	const float* begin = output_layer->cpu_data();
	const float* end = begin + OUTPUT_SIZE;
	pred = vector<float>(begin, end);

}


void CornersRegressor::regress(Pattern& pattern) {

	assert(test_net->phase() == caffe::TEST);

	// Input layer dimensions
	// First image
	Blob<float>* input_image_layer = test_net->input_blobs()[0];
	input_image_layer->Reshape(1, data_wrapper->get_channels(), data_wrapper->get_height(), data_wrapper->get_width());

	// Forward dimension change
	test_net->Reshape();

	// Wrap input layer to Mat's
	vector<Mat> image_channels;

	float* img_data = input_image_layer->mutable_cpu_data();

	for (int i = 0; i < data_wrapper->get_channels(); ++i) {
		Mat image_channel(data_wrapper->get_height(), data_wrapper->get_width(), CV_32FC1, img_data);
		image_channels.push_back(image_channel);
		img_data += data_wrapper->get_width() * data_wrapper->get_height();

	}

	split(pattern.image, image_channels);

	test_net->Forward();

	boost::shared_ptr<Blob<float>> output_layer = test_net->blob_by_name("output");

	const float* begin = output_layer->cpu_data();
	const float* end = begin + OUTPUT_SIZE;
	pattern.coords = vector<float>(begin, end);

}
