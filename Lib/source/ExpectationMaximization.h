/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianMixtureModel.h>

namespace gauss::gmm {
    std::vector<GaussianMixtureModel::Cluster> ExpectationMaximization(const TrainSet& train_set, const std::size_t& N_clusters, const GaussianMixtureModel::TrainInfo& info, double& likelihood);
}
