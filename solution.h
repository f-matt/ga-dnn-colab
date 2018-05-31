#ifndef SOLUTION_HPP
#define SOLUTION_HPP

#include "config.h"
#include "descriptors.h"
#include "corners-regressor.h"

#include <string>
#include <vector>

using namespace std;

class Solution {

public:
	Solution(boost::shared_ptr<CornersDataWrapper> &data_wrapper_ptr);

	Solution(boost::shared_ptr<CornersDataWrapper> &data_wrapper_ptr, const string& descriptor);

	virtual ~Solution();

    void evaluate();

    float get_fitness();

    void set_fitness(float fitness);

    bool operator< (const Solution &other) const;

    Solution& operator= (const Solution& other);

    void print();

    void crossover_conv(Solution& other);

    void crossover_fc(Solution& other);

    void mutate_conv();

    void mutate_fc();

    vector<string> get_layers();

    void set_layers(vector<string>);

    long get_weights();

    vector<float> get_output_vector();

    TopologyDescriptor get_descriptor();

    bool is_predicted();

    void set_predicted(bool m_predicted);

    bool is_valid();

private:
    void init();

    void init(const string& descriptor_str);

	void change_conv_descriptors(vector<ConvDescriptor> descriptors);

	void change_fc_descriptors(vector<FcDescriptor> descriptors);

	TopologyDescriptor descriptor;

	boost::shared_ptr<CornersDataWrapper> data_wrapper;

	float fitness;

	bool predicted;

	long n_weights;

};

#endif
