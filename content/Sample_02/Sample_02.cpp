#include "../GMM/GMM.h"
#include <iostream>
#include "../GMM/Utilities.h"
#include <time.h>


int main() {

	srand((unsigned int)time(0));
	size_t N_cluster = 6;

//build a random GMM
	Gaussian_Mixture_Model GMM_rand(N_cluster, 2);

// retrieve samples from the built GMM
	list<VectorXf> GMM_samples;
	GMM_rand.Get_samples(&GMM_samples, 500);
	print_Vectors("__Sampled_points", GMM_samples);

	MatrixXf params;
	GMM_rand.get_parameters(&params);
	print_Matrix("GMM_random", params);

// import the sampled points a training set
	Gaussian_Mixture_Model::Train_set train_set("__Sampled_points");

// build a new GMM, with the previous points passed for completing the EM algorithm
	Gaussian_Mixture_Model GMM_trained(train_set, N_cluster);
	GMM_trained.get_parameters(&params);
	print_Matrix("GMM_trained", params);

// classify the points in the training set
	list<VectorXf> GMM_trained_labels;
	GMM_trained.Classify(&GMM_trained_labels, GMM_samples);

// export the previous classification
	print_Vectors("__Sampled_points_classification", GMM_trained_labels);

//launch the Matlab example Main.m in the same folder to see the resuls

	system("pause");
	return 0;
}