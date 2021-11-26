/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianMixtureModel.h>
#include <GaussianDistributionFactory.h>

namespace gauss::gmm {
    class GaussianMixtureModelFactory
        : public RandomModelFactory<GaussianMixtureModel> {
    public:
        GaussianMixtureModelFactory(const std::size_t model_size, const std::size_t clusters);

        std::unique_ptr<GaussianMixtureModel> makeRandomModel() const override;

        void setClusters(std::size_t new_clusters) { clusters = new_clusters; }
        GaussianDistributionFactory& getClusterFactory() { return *cluster_factory.get(); };

    private:
        std::size_t clusters;
        std::unique_ptr<GaussianDistributionFactory> cluster_factory;
    };
} // namespace gauss
