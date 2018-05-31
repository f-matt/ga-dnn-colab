#ifndef CORNERS_REGRESSOR_H_
#define CORNERS_REGRESSOR_H_

#include "config.h"
#include "regressor.h"
#include "descriptors.h"
#include "corners-data-wrapper.h"


#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/timer.hpp>
#include <caffe/caffe.hpp>

using namespace caffe;
using namespace boost::filesystem;


class CornersRegressor : public Regressor<Pattern> {

public:
	CornersRegressor(const TopologyDescriptor &descriptor,
			boost::shared_ptr<CornersDataWrapper> &data_wrapper,
			int output_size,
			int batch_size,
			int max_epochs,
			int n_training_patterns,
			int n_test_patterns,
			int patience);

	CornersRegressor(boost::shared_ptr<CornersDataWrapper> &data_wrapper,
				int output_size,
				int batch_size,
				int max_epochs,
				int n_training_patterns,
				int n_test_patterns,
				int patience);

	CornersRegressor();

	virtual ~CornersRegressor();

	// Train the tracker
	void train();

	float train_with_batch(const vector<Pattern>& batch);

	void regress(const Pattern& pattern, vector<float> &pred, float &loss);

	void regress(Pattern& pattern);

private:
	void add_input_layer();

	void add_euclidean_loss_layer();

	boost::shared_ptr<CornersDataWrapper> data_wrapper;

};

#endif
