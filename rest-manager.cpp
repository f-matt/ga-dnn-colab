#include "rest-manager.h"
#include <jsoncpp/json/json.h>

RestManager::RestManager() {

}


RestManager::~RestManager() {

}


int writer(char *data, size_t size, size_t nmemb, std::string *writerData) {

	if(writerData == NULL)
		return 0;

	writerData->append(data, size*nmemb);

	return size * nmemb;

}


//size_t read_callback(void *dest, size_t size, size_t nmemb, void *userp) {
//
//	struct json_data *jdata = (struct json_data*) userp;
//	size_t buffer_size = size*nmemb;
//
//	if(jdata->sizeleft) {
//		/* copy as much as possible from the source to the destination */
//		size_t copy_this_much = jdata->sizeleft;
//		if(copy_this_much > buffer_size)
//			copy_this_much = buffer_size;
//		memcpy(dest, jdata->data, copy_this_much);
//
//		jdata->data += copy_this_much;
//		jdata->sizeleft -= copy_this_much;
//
//		return copy_this_much; /* we copied this many bytes */
//	}
//
//	return 0;
//}


FitnessRecord RestManager::get(const string& descriptor) const {
	FitnessRecord fr;

	try {
		CURL *curl;
		CURLcode res;
		string buffer;
		char error_buffer[CURL_ERROR_SIZE];

		curl_global_init(CURL_GLOBAL_ALL);
		curl = curl_easy_init();

		if (curl == NULL) {
			cerr << "Error creating curl connection." << endl;
			exit(EXIT_FAILURE);
		}

		res = curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buffer);
		if (res != CURLE_OK) {
			cerr << "Failed to set error buffer: " << res << endl;
			exit(EXIT_FAILURE);
		}

		res = curl_easy_setopt(curl, CURLOPT_URL, (GET_URL + descriptor).c_str());
		if (res != CURLE_OK) {
			cerr << "Failed to set write data " << error_buffer << endl;
			exit(EXIT_FAILURE);
		}

	
		res = curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writer);
		if (res != CURLE_OK) {
			cerr << "Failed to set write data " << error_buffer << endl;
			exit(EXIT_FAILURE);
		}

		res = curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
		if(res != CURLE_OK) {
			cerr << "Failed to set write data " << error_buffer << endl;
			exit(EXIT_FAILURE);
		}

		/* Perform the request, res will get the return code */
		res = curl_easy_perform(curl);

		curl_easy_cleanup(curl);

		/* Check for errors */
		if(res != CURLE_OK) {
			cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << endl;
			exit(EXIT_FAILURE);
		}

		std::stringstream ss;
		ss << buffer;

		if (!ss.str().empty()) {
			ptree root;

			read_json(ss, root);

			fr.descriptor = root.get<std::string>("descriptor");
			fr.training_input = root.get<std::string>("trainingInput");
			fr.fitness = root.get<float>("fitness");
			fr.n_weights = root.get<long>("nWeights");
		}

		curl_global_cleanup();
	} catch (std::exception& e) {
		std::cout << "[RestManager.get()] Exception: " << e.what() << std::endl;
	}

	return fr;


}
FitnessRecord RestManager::get_by_descriptor(const string& descriptor) const {

	FitnessRecord fr;

	try {
		fr = get(descriptor);

		if (fr.descriptor.empty()) {
			cout << "Empty response from REST. Falling back to local DB..." << endl;
		}

	} catch (std::exception& e) {
		std::cout << "[RestManager.get_by_descriptor()] Exception: " << e.what() << std::endl;
	}

	return fr;

}


void RestManager::insert(const FitnessRecord& fr) const {

	try {
		CURL *curl;
		CURLcode res;
		string buffer;
		char error_buffer[CURL_ERROR_SIZE];

		res = curl_global_init(CURL_GLOBAL_ALL);
		if (res != CURLE_OK) {
			cerr << "Error on curl init." << endl;
			exit(EXIT_FAILURE);
		}

		curl = curl_easy_init();
		if(curl == NULL) {
			cerr << "Error creating curl connection." << endl;
			exit(EXIT_FAILURE);
		}

		res = curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buffer);
		if (res != CURLE_OK) {
			cerr << "Failed to set error buffer: " << res << endl;
			exit(EXIT_FAILURE);
		}

		res = curl_easy_setopt(curl, CURLOPT_URL, POST_URL.c_str());
		if (res != CURLE_OK) {
			cerr << "Failed to set url: " << error_buffer << endl;
			exit(EXIT_FAILURE);
		}

		Json::Value val;
		val["descriptor"] = fr.descriptor;
		val["trainingInput"] = fr.training_input;
		val["fitness"] = fr.fitness;
		val["nWeights"] = (Json::Value::Int64) fr.n_weights;

		Json::FastWriter fast;

		string json_str = fast.write(val);

		res = curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str.c_str());
		if (res != CURLE_OK) {
			cerr << "Failed configuring post data: " << error_buffer << endl;
			exit(EXIT_FAILURE);
		}

		res = curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);
		if (res != CURLE_OK) {
			cerr << "Failed setting verbose opt: " << error_buffer << endl;
			exit(EXIT_FAILURE);
		}

		struct curl_slist *chunk = NULL;
		chunk = curl_slist_append(chunk, "Content-Type: application/json; charset=utf-8");
		res = curl_easy_setopt(curl, CURLOPT_HTTPHEADER, chunk);
		if (res != CURLE_OK) {
			cerr << "Failed setting HTTP header: " << error_buffer << endl;
			exit(EXIT_FAILURE);
		}

	    res = curl_easy_perform(curl);

	    if(res != CURLE_OK) {
	    	cerr << "curl_easy_perform() failed: " << error_buffer << endl;
	    	exit(EXIT_FAILURE);
	    }

	    curl_easy_cleanup(curl);
	    curl_global_cleanup();

	} catch (std::exception& e) {
		std::cout << "Exception: " << e.what() << std::endl;
	}

}
