/**
 * Author:    Andrea Casalino
 * Created:   03.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <header/GMM.h>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

string toJSON(const gmm::GMM& model) {
	auto printV = [](const gmm::V& v) {
		std::stringstream str;
		str << '[';
		str << v(0);
		for (std::size_t k = 1; k < (std::size_t)v.size(); ++k) {
			str << ',' << v(k);
		}
		str << ']';
		return str.str();
	};

	auto printCluster = [&printV](const gmm::GMMcluster& cl) {
		std::stringstream str;
		str << std::endl << '{';
		str << std::endl << "\"w\":" << cl.weight;
		str << std::endl << ",\"Mean\":" << printV(cl.Mean);
		str << std::endl << ",\"Covariance\":[";
		str << std::endl << printV(cl.Covariance.row(0));
		for (std::size_t k = 1; k < (std::size_t)cl.Covariance.rows(); ++k) {
			str << std::endl << ',' << printV(cl.Covariance.row(k));
		}
		str << std::endl << ']';
		str << std::endl << '}';
		return str.str();
	};

	const std::vector<gmm::GMMcluster>& clusters = model.getClusters();
	auto it = clusters.begin();

	std::stringstream str;
	str << '[';
	str << std::endl << printCluster(*it);
	++it;
	for (it; it != clusters.end(); ++it) {
		str << std::endl << ',' << printCluster(*it);
	}
	str << std::endl << ']';

	return str.str();
}

void print(const gmm::GMM& model, const std::string& file) {
	ofstream f(file);
	f << toJSON(model);
	f.close();
}
