/**
 * Author:    Andrea Casalino
 * Created:   03.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "Utils.h"
#include <header/KMeans.h>

void addSamplesFromCircle(std::list<gmm::V>& samples, const double& x, const double& y, const double& ray, const std::size_t& N_sample);

int main() {
// sample some points in different region of the spaces
	list<gmm::V> Samples;
	std::size_t sectors = 5;
	double angle = 0.0;
	for (std::size_t k = 0; k < sectors; ++k) {
		addSamplesFromCircle(Samples, 6.0 * cos(angle), 6.0 * sin(angle), 2.0, 50);
		angle += 2.0 * static_cast<double>(3.14159) / static_cast<double>(sectors);
	}
	gmm::TrainSet set(Samples);

// perform K_means clustering
	vector<list<const gmm::V*>> clusters;
	gmm::kMeansClustering(clusters, set, sectors);

// save data for posterior plotting
	ofstream f("K_means_clustering");
	auto it_s = clusters.front().begin();
	gmm::V temp;
	std::size_t id = 0;
	for (auto it_cl = clusters.begin(); it_cl != clusters.end(); ++it_cl) {
		for (it_s = it_cl->begin(); it_s != it_cl->end(); ++it_s)
			f << id << " " << (*it_s)->transpose() << endl;
		++id;
	}
	f.close();

//use the python script Visualize04.py to see the results

	return EXIT_SUCCESS;
}

void addSamplesFromCircle(std::list<gmm::V>& samples, const double& x, const double& y, const double& ray, const std::size_t& N_sample) {
	double r;
	double teta;
	for (std::size_t k = 0; k < N_sample; ++k) {
		r = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX))* ray;
		teta = 2.0 * static_cast<double>(3.14159)* static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		samples.emplace_back(2);
		samples.back()(0) = x + r * cos(teta);
		samples.back()(1) = y + r * sin(teta);
	}
}
