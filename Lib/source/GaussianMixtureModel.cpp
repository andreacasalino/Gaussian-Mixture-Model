/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianMixtureModel.h>
#include <GaussianMixtureModelSampler.h>
#include <Error.h>
#include "EvaluateLogDensity.h"
#include <ExpectationMaximization.h>

namespace gauss::gmm {
    std::vector<GaussianMixtureModel::Cluster> check_and_make_clusters(const std::vector<GaussianMixtureModel::Cluster>& clusters) {
        std::vector<GaussianMixtureModel::Cluster> result = clusters;
        if (result.empty()) {
            throw Error("Empy clusters for GaussianMixtureModel");
        }
        double weight_sum = 0.0;
        std::size_t state_size = result.front().distribution.getMean().size();
        for (const auto& cluster : result) {
            if (cluster.weight < 0) {
                throw Error("Negative weight for GaussianMixtureModel");
            }
            if (state_size != cluster.distribution.getMean().size()) {
                throw Error("Invalid cluster set");
            }
            weight_sum += cluster.weight;
        }
        for (auto& cluster : result) {
            cluster.weight *= 1.0 / weight_sum;
        }
        return result;
    }
    GaussianMixtureModel::GaussianMixtureModel(const std::vector<Cluster>& clusters)
        : clusters(check_and_make_clusters(clusters)) {
    }

    std::unique_ptr<GaussianMixtureModel> GaussianMixtureModel::fitOptimalModel(const TrainSet& train_set, const std::vector<std::size_t>& N_clusters_to_try, const std::size_t& Iterations) {
        if (N_clusters_to_try.empty()) {
            throw Error("No clusters to try");
        }
        TrainInfo info;
        info.maxIterations = Iterations;

        double best_likelihood;
        std::unique_ptr<std::vector<GaussianMixtureModel::Cluster>> best_model =
            std::make_unique<std::vector<GaussianMixtureModel::Cluster>>(ExpectationMaximization(train_set, N_clusters_to_try.front(), info, &best_likelihood));
        for (std::size_t k = 1; k < N_clusters_to_try.size(); ++k) {
            double att_likelihood;
            std::unique_ptr<std::vector<GaussianMixtureModel::Cluster>> att_model =
                std::make_unique<std::vector<GaussianMixtureModel::Cluster>>(ExpectationMaximization(train_set, N_clusters_to_try.front(), info, &att_likelihood));
            if (att_likelihood > best_likelihood) {
                best_likelihood = att_likelihood;
                best_model = std::move(att_model);
            }
        }
        return std::make_unique<GaussianMixtureModel>(*best_model);
    }

    Eigen::VectorXd GaussianMixtureModel::Classify(const Eigen::VectorXd& point) const {
        Eigen::VectorXd result(clusters.size());
        for (Eigen::Index k = 0; k < result.size(); ++k) {
            result(k) = exp(log(clusters[k].weight) + clusters[k].distribution.evaluateLogDensity(point));
        }
        result *= (1.0 / result.sum());
        return result;
    }

    std::vector<Eigen::VectorXd>
        GaussianMixtureModel::drawSamples(const std::size_t samples) const {
        GaussianMixtureModelSampler sampler(*this);
        std::vector<Eigen::VectorXd> result;
        result.reserve(samples);
        for (std::size_t s = 0; s < samples; ++s) {
            result.push_back(sampler.getSample());
        }
        return result;
    }

    double GaussianMixtureModel::evaluateLogDensity(const Eigen::VectorXd& point) const {
        std::vector<double> weights;
        weights.reserve(clusters.size());
        std::vector<const GaussianDistribution*> distributions;
        distributions.reserve(clusters.size());
        for (const auto&  cluster : clusters) {
            weights.push_back(cluster.weight);
            distributions.push_back(&cluster.distribution);
        }
        return evaluate_log_density(point, weights, distributions);
    }
}
