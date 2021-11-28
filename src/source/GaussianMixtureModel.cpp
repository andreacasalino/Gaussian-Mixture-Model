/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianMixtureModel/GaussianMixtureModel.h>
#include <GaussianMixtureModel/GaussianMixtureModelSampler.h>
#include <GaussianMixtureModel/Error.h>
#include "EvaluateLogDensity.h"
#include <GaussianMixtureModel/ExpectationMaximization.h>
#include <Eigen/Dense>

namespace gauss::gmm {
    namespace {
        std::vector<Cluster> check_and_make_clusters(const std::vector<Cluster>& clusters) {
            std::vector<Cluster> result = clusters;
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
        std::unique_ptr<std::vector<Cluster>> best_model =
            std::make_unique<std::vector<Cluster>>(ExpectationMaximization(train_set, N_clusters_to_try.front(), info, &best_likelihood));
        for (std::size_t k = 1; k < N_clusters_to_try.size(); ++k) {
            double att_likelihood;
            std::unique_ptr<std::vector<Cluster>> att_model =
                std::make_unique<std::vector<Cluster>>(ExpectationMaximization(train_set, N_clusters_to_try.front(), info, &att_likelihood));
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

    double GaussianMixtureModel::evaluateKullbackLeiblerDivergence(
        const GaussianMixtureModel& other) const {
        if (clusters.front().distribution.getMean().size() != other.getClusters().front().distribution.getMean().size()) {
            throw Error("The 2 GaussianMixtureModel are not comparable");
        }
        auto samples = drawSamples(monte_carlo_trials);

        double div = 0.f;
        for (const auto& sample : samples) {
            div += evaluateLogDensity(sample);
            div -= other.evaluateLogDensity(sample);
        }
        div *= 1.0 / static_cast<double>(samples.size());
        return div;
    }

    namespace {
        double tOperator(const Cluster& f, const Cluster& g) {
            double temp = -f.distribution.getMean().size() * log(2.0 * PI_GREEK);
            Eigen::MatrixXd S = f.distribution.getCovariance();
            S += g.distribution.getCovariance();
            temp -= log(S.determinant());
            Eigen::VectorXd Delta = g.distribution.getMean() - f.distribution.getMean();
            temp -= Delta.transpose() * computeCovarianceInvert(S) * Delta;
            temp *= 0.5;
            return exp(temp);
        }
    }

    std::pair<double, double> GaussianMixtureModel::estimateKullbackLeiblerDivergence(const GaussianMixtureModel& other) const {
        if (clusters.front().distribution.getMean().size() != other.getClusters().front().distribution.getMean().size()) {
            throw Error("The 2 GaussianMixtureModel are not comparable");
        }
        Eigen::MatrixXd Divergences(clusters.size(), other.clusters.size());
        Eigen::MatrixXd t(clusters.size(), other.clusters.size());
        Eigen::MatrixXd z(clusters.size(), clusters.size());
        std::size_t a, A = clusters.size(), b, B = other.clusters.size();
        for (a = 0; a < A; ++a) {
            for (b = 0; b < B; ++b) {
                Divergences(a, b) = clusters[a].distribution.evaluateKullbackLeiblerDivergence(other.clusters[b].distribution); 
                t(a, b) = tOperator(clusters[a], other.clusters[b]);
            }
        }
        for (a = 0; a < A; ++a) {
            for (b = 0; b < A; ++b) {
                z(a, b) = tOperator(clusters[a], clusters[b]);
            }
        }

        // <lower , upper> bound
        std::pair<double, double> bound = std::make_pair<double, double>(0.0, 0.0);
        std::size_t a2;
        double temp = 0.f, temp2, temp3;
        for (a = 0; a < A; ++a) {
            temp2 = 0.f;
            for (a2 = 0; a2 < A; ++a2)  temp2 += clusters[a2].weight * z(a, a2);
            temp2 = log(temp2);
            temp3 = 0.f;
            for (b = 0; b < B; ++b)  temp3 += other.clusters[b].weight * exp(-Divergences(a, b));
            temp3 = log(temp3);
            bound.second += clusters[a].weight * (temp2 - temp3);

            temp2 = 0.f;
            for (a2 = 0; a2 < A; ++a2)  temp2 += clusters[a2].weight * exp(-Divergences(a, b));
            temp2 = log(temp2);
            temp3 = 0.f;
            for (b = 0; b < B; ++b)  temp3 += other.clusters[b].weight * t(a, b);
            temp3 = log(temp3);
            bound.first += clusters[a].weight * (temp2 - temp3);

            temp += clusters[a].weight * 0.5 * log(pow(2.f * PI_GREEK * 2.71828, (double)clusters[a].distribution.getMean().size()) * abs(clusters[a].distribution.getCovarianceDeterminant()) );
        }

        bound.second += temp;
        bound.first -= temp;
        return bound;
    }
}
