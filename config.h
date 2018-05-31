#ifndef CONFIG_H_
#define CONFIG_H_

// #define CPU_ONLY
#define USE_OPENCV

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

const int OUTPUT_SIZE = 8;

const int BATCH_SIZE = 32;

const int MAX_EPOCHS = 100;

const int TRAIN_PATTERNS = 1024;

const int TEST_PATTERNS = 256;

const int PATIENCE = 10;


/*
 * OpenCV images (reference + current image) + coorners coords
 */
struct Pattern {
	Mat image;
	vector<float> coords;
};


/*
 * Fitness prediction pattern
 */
struct FitnessPattern {
	int id;
	string descriptor;
	vector<float> input;
	float fitness;
	long n_weights;
};


/*
 * Fitness record, used for storage in db
 */
struct FitnessRecord {
	int id;
	string descriptor;
	string training_input;
	float fitness;
	long n_weights;
};


#endif
