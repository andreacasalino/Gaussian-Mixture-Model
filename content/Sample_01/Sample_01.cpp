#include "../GMM/GMM.h"
#include <iostream>
#include "../GMM/Utilities.h"

int main() {

	size_t N_sample = 500;
	size_t N_clusters = 5;

//produce a list of random samples
	VectorXf Cube_Sizes(2);
	Cube_Sizes << 1.f, 1.f;
	list<VectorXf> samples;

	get_random_samples(&samples, Cube_Sizes, N_sample);
	print_Vectors("__Sampled_points", samples);

// apply k means to the sampled points
	list<list<VectorXf*>> clusters;
	Gaussian_Mixture_Model::K_means::do_clustering(&clusters, samples, N_clusters);

	//print the samples, with labels specifying corresponding assigned the cluster
	print_clusters("__Clustered_points", clusters);

//launch the Matlab example Main.m in the same folder to see the resuls

	system("pause");
	return 0;
}
