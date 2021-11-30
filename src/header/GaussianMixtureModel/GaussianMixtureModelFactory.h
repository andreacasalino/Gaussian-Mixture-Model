/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianMixtureModel/GaussianMixtureModel.h>
#include <GaussianUtils/GaussianDistributionFactory.h>

namespace gauss::gmm {
    class GaussianMixtureModelFactory
        : public RandomModelFactory<GaussianMixtureModel>
        , public StateSpaceSizeAware {
    public:
        GaussianMixtureModelFactory(const std::size_t model_size, const std::size_t clusters);

        std::size_t getStateSpaceSize() const override { return cluster_factory.getStateSpaceSize(); }

        std::unique_ptr<GaussianMixtureModel> makeRandomModel() const override;

        void setClusters(std::size_t new_clusters);
        GaussianDistributionFactory& accessClusterFactory() { return cluster_factory; };

    private:
        std::size_t clusters;
        GaussianDistributionFactory cluster_factory;
    };
} // namespace gauss
