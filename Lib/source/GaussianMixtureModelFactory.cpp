/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianMixtureModelFactory.h>

namespace gauss::gmm {
    GaussianMixtureModelFactory::GaussianMixtureModelFactory(const std::size_t model_size, const std::size_t clusters)
        : clusters(clusters) {
        cluster_factory = std::make_unique<GaussianDistributionFactory>(model_size);
    }


}
