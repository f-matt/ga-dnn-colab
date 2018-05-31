#ifndef REST_MANAGER_H_
#define REST_MANAGER_H_

#include "config.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string.hpp>

#include <curl/curl.h>

#include <iostream>
#include <string>
#include <sstream>

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;

using namespace std;

//const string GET_URL = "http://ga-dnn.herokuapp.com/get?descriptor=";
//const string POST_URL = "http://ga-dnn.herokuapp.com/post";

const string GET_URL = "http://fmatt.pythonanywhere.com/get-ar/";
const string POST_URL = "http://fmatt.pythonanywhere.com/post-ar";

class RestManager {
public:
	RestManager();

	virtual ~RestManager();

	FitnessRecord get(const string& descriptor) const;

	FitnessRecord get_by_descriptor(const string& descriptor) const;

	void insert(const FitnessRecord& fr) const;

};


struct json_data {
	const char* data;
	size_t sizeleft;
};

#endif
