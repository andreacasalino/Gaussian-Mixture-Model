/**
 * Author:    Andrea Casalino
 * Created:   03.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianMixtureModel/GaussianMixtureModel.h>
#include <GaussianMixtureModel/GaussianMixtureModelFactory.h>
#include <GaussianMixtureModel/ExpectationMaximization.h>
#include "Utils.h"

std::pair<double, double> getSupport(const std::vector<double>& values);

std::unique_ptr<gauss::gmm::GaussianMixtureModel> getSimilarModel(const gauss::gmm::GaussianMixtureModel& model_to_emulate);

///////////////////////////////////////////////
// Divergences of high dimensions GMM models //
///////////////////////////////////////////////
int main() {
//sample a random high dimensional GMM model
	std::size_t N_clusters = 5;
	std::size_t Dimension = 10;
	gauss::gmm::GaussianMixtureModelFactory model_factory(Dimension, N_clusters);
	auto model = model_factory.makeRandomModel();

	double divergence_learnt;
	std::vector<double> divergences_similar;
	std::vector<double> divergences_purerandom;

//fit a model considering samples drawn from the initial model
	gauss::TrainSet train_set(model->drawSamples(2000));
	gauss::gmm::GaussianMixtureModel learnt_model(gauss::gmm::ExpectationMaximization(train_set, N_clusters));

//compute the Divergence of the two models
	divergence_learnt = model->evaluateKullbackLeiblerDivergence(learnt_model);

//compute some similar models to the initial sampled one and evaluate their divergences w.r.t. the orginal model
	std::size_t model_samples = 7;
	for (std::size_t k = 0; k < model_samples; ++k) {
		auto similar = getSimilarModel(*model);
		divergences_similar.push_back(model->evaluateKullbackLeiblerDivergence(*similar));
	}

//compute some pure random models and evaluate their divergences w.r.t. the orginal model
	for (std::size_t k = 0; k < model_samples; k++) {
		auto random_model = model_factory.makeRandomModel();
		divergences_purerandom.push_back(model->evaluateKullbackLeiblerDivergence(*random_model));
	}

//compare the divergences w.r.t: the learnt model, the similar models and the pure randomic ones
	std::cout << "divergence with the learnt model " << divergence_learnt << std::endl;

	auto bound_similar = getSupport(divergences_similar);
	std::cout << "divergence with similar models             [min,max] :  [" << bound_similar.first << "  ,  " << bound_similar.second << " ]" << std::endl;

	auto bound_random = getSupport(divergences_purerandom);
	std::cout << "divergence with pure random models [min,max] :  [" << bound_random.first << "  ,  " << bound_random.second << " ]" << std::endl;

	return EXIT_SUCCESS;
}

std::pair<double, double> getSupport(const std::vector<double>& values) {
	auto it = values.begin();
	std::pair<double, double> bound = std::make_pair(*it, *it);
	it++;
	for (it; it != values.end(); it++) {
		if (*it > bound.second) bound.second = *it;
		if (*it < bound.first) bound.first = *it;
	}
	return bound;
}

std::unique_ptr<gauss::gmm::GaussianMixtureModel> getSimilarModel(const gauss::gmm::GaussianMixtureModel& model_to_emulate) {
	const auto& old_clusters = model_to_emulate.getClusters();
	std::vector<gauss::gmm::Cluster> new_clusters;
	new_clusters.reserve(old_clusters.size());
	for (const auto& old_cluster : old_clusters) {
		double new_weight = old_cluster.weight + 0.02 * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		auto new_mean = old_cluster.distribution->getMean();
		{
			Eigen::VectorXd delta(new_mean.size());
			delta.setRandom();
			delta *= 0.1;
			new_mean += delta;
		}
		new_clusters.emplace_back();
		new_clusters.back().weight = new_weight;
		new_clusters.back().distribution = std::make_unique<gauss::GaussianDistribution>(new_mean, old_cluster.distribution->getCovariance());
	}
	return std::make_unique<gauss::gmm::GaussianMixtureModel>(new_clusters);
}