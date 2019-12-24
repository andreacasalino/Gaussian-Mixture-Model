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

void get_support(float* max, float* min, const list<float>& values);

void get_similar_models(list<Gaussian_Mixture_Model>* similar , const Gaussian_Mixture_Model& model_to_modifiy, const size_t& N_models);

int main() {

//sample a random high dimensional GMM model
	size_t N_clusters = 5;
	size_t Dimension = 10;
	Gaussian_Mixture_Model model(N_clusters, Dimension);

//fit a model considering samples drawn from the model
	list<VectorXf>  train_set;
	size_t	train_set_size = 1000;
	model.Get_samples(&train_set, train_set_size);
	Gaussian_Mixture_Model::Train_set set(train_set);
	Gaussian_Mixture_Model learnt_model(N_clusters, set);

//compute the Divergence of the two models
	float div_learnt = model.Get_KULLBACK_LEIBLER_divergence_MonteCarlo(learnt_model);

//compute some similar models to the initial sampled one and evaluate their divergences w.r.t. the orginal model
	size_t model_samples = 7;
	list<float> differences_with_similar_models;
	list<Gaussian_Mixture_Model> similar_models;
	get_similar_models(&similar_models, model, model_samples);
	for (size_t k = 0; k < model_samples; k++) {
		differences_with_similar_models.push_back(model.Get_KULLBACK_LEIBLER_divergence_MonteCarlo(similar_models.front()));
		//differences_with_similar_models.push_back(model.Get_KULLBACK_LEIBLER_divergence_MonteCarlo(model));
		similar_models.pop_front();
	}

//compute some more random models and evaluate their divergences w.r.t. the orginal model
	list<float> differences_with_pure_random_models;
	for (size_t k = 0; k < model_samples; k++) {
		Gaussian_Mixture_Model model_temp(N_clusters, Dimension);
		differences_with_pure_random_models.push_back(model.Get_KULLBACK_LEIBLER_divergence_MonteCarlo(model_temp));
	}

//compare the divergences w.r.t: the learnt model, the similar models and the pure randomic ones
	cout << "divergence with the learnt model " << div_learnt << endl;
	float div_similar[2];
	get_support(&div_similar[1], &div_similar[0], differences_with_similar_models);
	cout << "divergence with similar models             [min,max] :  [" << div_similar[0] << "  ,  " << div_similar[1] << " ]" << endl;
	float div_pure_random[2];
	get_support(&div_pure_random[1], &div_pure_random[0], differences_with_pure_random_models);
	cout << "divergence with pure random models [min,max] :  [" << div_pure_random[0] << "  ,  " << div_pure_random[1] << " ]" << endl;


	return 0;
}

void get_support(float* max, float* min, const list<float>& values) {

	auto it = values.begin();
	*min = *it;
	*max = *it;
	it++;
	for (it; it != values.end(); it++) {
		if (*it > *max) *max = *it;
		if (*it < *min) *min = *it;
	}

}

void get_similar_models(list<Gaussian_Mixture_Model>* similar, const Gaussian_Mixture_Model& model_to_modifiy, const size_t& N_models) {

	VectorXf min_V, max_V;
	MatrixXf vals = model_to_modifiy.get_parameters_as_matrix();
	MatrixXf vals_similar;
	size_t N_cluster = vals.rows() / (2 + vals.cols());
	size_t x = vals.cols();

	min_V = vals.row(1);
	max_V = min_V;
	size_t c, C =vals.cols();
	for (size_t k = 1; k < N_cluster; k++) {
		for (c = 0; c < C; c++) {
			if (vals.row(k*(2+x) + 1)(c) < min_V(c)) min_V(c) = vals.row(k*(2 + x) + 1)(c);
			if (vals.row(k*(2 + x) + 1)(c) > max_V(c)) max_V(c) = vals.row(k*(2 + x) + 1)(c);
		}
	}

	VectorXf Delta = max_V - min_V;
	Delta *=  0.1f;

	for (size_t m = 0; m < N_models; m++) {
		vals_similar = vals;
		for (size_t k = 0; k < N_cluster; k++) {
			for (c = 0; c < C; c++)
				vals_similar.row(k*(2 + x) + 1)(c) += Delta(c) * (float)rand() / (float)RAND_MAX;
		}
		similar->push_back(Gaussian_Mixture_Model(vals_similar));
	}

}