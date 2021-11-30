/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianMixtureModel/GaussianMixtureModel.h>

namespace gauss::gmm {
	/** @brief The GMM is be built using the passed train set using the expectation maximization algorithm.
	 * The initial guess used to cluster the samples, might be obtained using the K means or directly specified, according
	 * to the values put in TrainInfo.
	 * @param[in] the number of clusters to consider for training
	 * @param[in] the training set to use
	 * @param[in] the information used by the training process
	 */
	struct TrainInfo {
		// The maximum number of iterations considered by the expectation maximization algorithm.
		std::size_t maxIterations = 1000;
		// when passed empty is ingored and the K means is used for building the initial guess
		std::vector<std::size_t> initialLabeling = {};
	};

	std::vector<Cluster> ExpectationMaximization(const TrainSet& train_set, const std::size_t& N_clusters, const TrainInfo& info = TrainInfo{}, double* likelihood = nullptr);
}
