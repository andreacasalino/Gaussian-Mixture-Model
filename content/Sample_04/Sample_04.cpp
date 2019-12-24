/**
 * Author:    Andrea Casalino
 * Created:   03.12.2019
*
* report any bug to andrecasa91@gmail.com.
 **/

#include "../GMM/GMM.h"
#include <iostream>
#include <fstream>
using namespace std;
using namespace Eigen;

void get_samples_in_circle(list<VectorXf>* samples , const float& x, const float& y, const float& ray, const size_t& N_sample) {

	float r;
	float teta;
	for (size_t k = 0; k < N_sample; k++) {
		r = (float)rand() / (float)RAND_MAX * ray;
		teta = (float)rand() / (float)RAND_MAX * 2.f * (float)M_PI;
		samples->push_back(VectorXf(2));
		samples->back()(0) = x + r * cosf(teta);
		samples->back()(1) = y + r * sinf(teta);
	}

}

int main() {

// sample some points in different region of the spaces
	list<VectorXf> Samples;
	size_t sectors = 4;
	float angle = 0.f;
	for (size_t k = 0; k < sectors; k++) {
		get_samples_in_circle(&Samples, 6.f * cosf(angle), 6.f * sinf(angle), 2.f, 20);
		angle += 2.f *(float)M_PI / (float)sectors;
	}
	Gaussian_Mixture_Model::Train_set set(Samples);

// perform K_means clustering
	list<list<const VectorXf*>> clusters;
	Gaussian_Mixture_Model::K_means_clustering(&clusters, set, sectors);

// save data for posterior plotting
	ofstream f("../Result_visualization/K_means_clustering");
	auto it_s = clusters.front().begin();
	VectorXf temp;
	size_t id = 0;
	for (auto it_cl = clusters.begin(); it_cl != clusters.end(); it_cl++) {
		for (it_s = it_cl->begin(); it_s != it_cl->end(); it_s++)
			f << id << " " << (*it_s)->transpose() << endl;
		id++;
	}
	f.close();

	return 0;
}
