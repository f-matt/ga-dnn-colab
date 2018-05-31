#include "corners-data-wrapper.h"

#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>

const int IMG_WIDTH = 100;
const int IMG_HEIGHT = 100;
const int IMG_CHANNELS = 3;
const int IMG_TYPE = CV_32FC3;

CornersDataWrapper::CornersDataWrapper() : train_file("data/ar-tracking/train.csv"),
		test_file("data/ar-tracking/test.csv"),
		height(IMG_HEIGHT),
		width(IMG_WIDTH),
		channels(IMG_CHANNELS),
		type(IMG_TYPE) {

}


CornersDataWrapper::~CornersDataWrapper() {

}


vector<Pattern> CornersDataWrapper::load_patterns(const string& filename, const int n_patterns) const {

	vector<Pattern> patterns(n_patterns);

	string line;

	ifstream myfile(filename);

	if (myfile.is_open()) {

		for (int i = 0; i < n_patterns; ++i) {

			// Read line from file
			getline(myfile, line);

			vector<string> tokens;

			boost::split(tokens, line, boost::is_any_of(";"));

			patterns[i].image = imread(tokens[0], CV_LOAD_IMAGE_COLOR);
			patterns[i].image.convertTo(patterns[i].image, CV_32FC3, 1.0 /255, 0);

			patterns[i].coords = vector<float>(8);

			for (int j = 0; j < 8; ++j)
				patterns[i].coords[j] = (stof(tokens[j + 1]) - CornersDataWrapper::OUTPUT_NORM_FACTOR) / CornersDataWrapper::OUTPUT_NORM_FACTOR;

		}

		myfile.close();

	} else {
		cerr << "Error opening file: " << filename << endl;
		exit(EXIT_FAILURE);
	}

	return patterns;

}


pair<vector<Pattern>, vector<Pattern>> CornersDataWrapper::load_train_test_data(unsigned int n_train_patterns, unsigned int n_test_patterns) {

	pair<vector<Pattern>, vector<Pattern>> train_test_patterns;

	vector<Pattern> train_patterns = load_patterns(train_file, n_train_patterns);
	vector<Pattern> test_patterns = load_patterns(test_file, n_test_patterns);

	// Check image shape
	if ((width != train_patterns[0].image.cols) || (height != train_patterns[0].image.rows) || (channels != train_patterns[0].image.channels())) {
		cerr << "Incorrect image dimensions. Found: " << train_patterns[0].image.cols << " " <<  train_patterns[0].image.rows << " " << train_patterns[0].image.channels() <<
				"while epecting " << width << " " << height << " " << channels << endl;
		exit(EXIT_FAILURE);
	}

	Mat mean_image = Mat::zeros(height, width, type);

	for (Pattern& p : train_patterns) {
		mean_image += p.image;
	}

	mean_image /= train_patterns.size();

	for (Pattern& p : train_patterns) {
		p.image -= mean_image;
	}

	for (Pattern& p : test_patterns) {
		p.image -= mean_image;
	}

	train_test_patterns.first = train_patterns;
	train_test_patterns.second = test_patterns;

	return train_test_patterns;

}


const int& CornersDataWrapper::get_height() const {
	return height;
}


const int& CornersDataWrapper::get_width() const {
	return width;
}


const int& CornersDataWrapper::get_channels() const {
	return channels;
}


const int& CornersDataWrapper::get_type() const {
	return type;
}
