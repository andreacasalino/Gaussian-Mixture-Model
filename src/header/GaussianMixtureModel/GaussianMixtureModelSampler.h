/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianUtils/GaussianDistributionSampler.h>
#include <GaussianMixtureModel/GaussianMixtureModel.h>

namespace gauss::gmm {
    class GaussianMixtureModelSampler
        : public StateSpaceSizeAware {
    public:
        GaussianMixtureModelSampler(const GaussianMixtureModel& distribution);

        std::size_t getStateSpaceSize() const override { return clusters_samplers.front().getStateSpaceSize(); }

        Eigen::VectorXd getSample() const;

    private:
        std::vector<double> clusters_weights;
        std::vector<GaussianDistributionSampler> clusters_samplers;
        mutable std::uniform_real_distribution<double> unif_iso;
        mutable std::default_random_engine generator;
    };
} // namespace gauss
