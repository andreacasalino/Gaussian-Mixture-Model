/**
 * Author:    Andrea Casalino
 * Created:   03.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianMixtureModel/GaussianMixtureModel.h>
#include <iostream>
#include <fstream>
#include <sstream>

std::string toJSON(const Eigen::VectorXd& vector) {
	std::stringstream str;
	str << '[';
	str << vector(0);
	for (Eigen::Index k = 1; k < vector.size(); ++k) {
		str << ',' << vector(k);
	}
	str << ']';
	return str.str();
};

std::string toJSON(const gauss::gmm::Cluster& cl) {
	std::stringstream str;
	str << std::endl << '{';
	str << std::endl << "\"w\":" << cl.weight;
	str << std::endl << ",\"Mean\":" << toJSON(cl.distribution.getMean());
	auto covariance = cl.distribution.getCovariance();
	str << std::endl << ",\"Covariance\":[";
	str << std::endl << toJSON(covariance.row(0));
	for (Eigen::Index k = 1; k < covariance.rows(); ++k) {
		str << std::endl << ',' << toJSON(covariance.row(k));
	}
	str << std::endl << ']';
	str << std::endl << '}';
	return str.str();
};

std::string toJSON(const gauss::gmm::GaussianMixtureModel& model) {
	const auto& clusters = model.getClusters();
	auto it = clusters.begin();
	std::stringstream str;
	str << '[';
	str << std::endl << toJSON(*it);
	++it;
	for (it; it != clusters.end(); ++it) {
		str << std::endl << ',' << toJSON(*it);
	}
	str << std::endl << ']';
	return str.str();
}

void print(const gauss::gmm::GaussianMixtureModel& model, const std::string& file) {
	std::ofstream f(file);
	f << toJSON(model);
	f.close();
}
