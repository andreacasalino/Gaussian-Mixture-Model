/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianMixtureModel.h>
#include <GaussianMixtureModelSampler.h>

namespace gauss::gmm {
    Eigen::VectorXd GaussianMixtureModel::Classify(const Eigen::VectorXd& point) const {
        Eigen::VectorXd result(clusters.size());
        for (Eigen::Index k = 0; k < result.size(); ++k) {
            result(k) = exp(clusters[k].getWeightLog() + clusters[k].distribution->evaluateLogDensity(point));
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
        double den = 0.0;
        for (const auto& cluster : clusters) {
            den += exp(cluster.getWeightLog() + cluster.distribution->evaluateLogDensity(point));
        }
        den = log(den);
        return den;
    }
}
