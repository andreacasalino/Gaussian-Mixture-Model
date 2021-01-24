/**
 * Author:    Andrea Casalino
 * Created:   03.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "Utils.h"

std::pair<double, double> getSupport(const vector<double>& values);

std::unique_ptr<gmm::GMM> getSimilarModel(const gmm::GMM& model_to_emulate);

int main() {

//sample a random high dimensional GMM model
	std::size_t N_clusters = 5;
	std::size_t Dimension = 10;
	gmm::GMM model(N_clusters, Dimension);

	double divergence_learnt;
	vector<double> divergences_similar;
	vector<double> divergences_purerandom;

//fit a model considering samples drawn from the initial model
	list<gmm::V> train_set = model.drawSamples(2000);
	gmm::GMM learnt_model(N_clusters, gmm::TrainSet(train_set));

//compute the Divergence of the two models
	divergence_learnt = model.getKullbackLeiblerDiergenceMonteCarlo(learnt_model);

//compute some similar models to the initial sampled one and evaluate their divergences w.r.t. the orginal model
	std::size_t model_samples = 7;
	for (std::size_t k = 0; k < model_samples; ++k) {
		auto similar = getSimilarModel(model);
		divergences_similar.push_back(model.getKullbackLeiblerDiergenceMonteCarlo(*similar));
	}

//compute some more random models and evaluate their divergences w.r.t. the orginal model
	for (std::size_t k = 0; k < model_samples; k++) {
		gmm::GMM model_temp(N_clusters, Dimension);
		divergences_purerandom.push_back(model.getKullbackLeiblerDiergenceMonteCarlo(model_temp));
	}

//compare the divergences w.r.t: the learnt model, the similar models and the pure randomic ones
	cout << "divergence with the learnt model " << divergence_learnt << endl;

	auto bound_similar = getSupport(divergences_similar);
	cout << "divergence with similar models             [min,max] :  [" << bound_similar.first << "  ,  " << bound_similar.second << " ]" << endl;

	auto bound_random = getSupport(divergences_purerandom);
	cout << "divergence with pure random models [min,max] :  [" << bound_random.first << "  ,  " << bound_random.second << " ]" << endl;

	return EXIT_SUCCESS;
}

std::pair<double, double> getSupport(const vector<double>& values) {
	auto it = values.begin();
	std::pair<double, double> bound = std::make_pair(*it, *it);
	it++;
	for (it; it != values.end(); it++) {
		if (*it > bound.second) bound.second = *it;
		if (*it < bound.first) bound.first = *it;
	}
	return bound;
}

std::unique_ptr<gmm::GMM> getSimilarModel(const gmm::GMM& model_to_emulate) {
	gmm::V min_V, max_V;
	gmm::M vals = model_to_emulate.getMatrixParameters();
	gmm::M vals_similar;
	std::size_t N_cluster = vals.rows() / (2 + vals.cols());
	std::size_t x = vals.cols();

	min_V = vals.row(1);
	max_V = min_V;
	std::size_t c, C =vals.cols();
	for (std::size_t k = 1; k < N_cluster; k++) {
		for (c = 0; c < C; c++) {
			if (vals.row(k*(2+x) + 1)(c) < min_V(c)) min_V(c) = vals.row(k*(2 + x) + 1)(c);
			if (vals.row(k*(2 + x) + 1)(c) > max_V(c)) max_V(c) = vals.row(k*(2 + x) + 1)(c);
		}
	}

	gmm::V Delta = max_V - min_V;
	Delta *=  0.1;

	vals_similar = vals;
	for (std::size_t k = 0; k < N_cluster; k++) {
		for (c = 0; c < C; c++)
			vals_similar.row(k * (2 + x) + 1)(c) += Delta(c) * (float)rand() / (float)RAND_MAX;
	}
	return std::make_unique<gmm::GMM>(vals_similar);
}