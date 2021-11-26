/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianMixtureModelFactory.h>
#include <Error.h>

namespace gauss::gmm {
    GaussianMixtureModelFactory::GaussianMixtureModelFactory(const std::size_t model_size, const std::size_t clusters)
        : clusters(clusters)
        , cluster_factory(model_size) {
        if (0 == clusters) {
            throw Error("0 clusters not admitted");
        }
    }

    std::unique_ptr<GaussianMixtureModel> GaussianMixtureModelFactory::makeRandomModel() const {
        std::vector<GaussianMixtureModel::Cluster> clusters_data;
        clusters_data.reserve(clusters);
        Eigen::VectorXd weights(clusters);
        weights.setRandom();
        {
            double sum = 0.0;
            for (Eigen::Index i = 0; i < weights.size(); ++i) {
                weights(i) = abs(weights(i));
                sum += weights(i);
            }
            weights *= 1.0 / sum;
        }
        for (std::size_t c = 0; c < clusters; ++c) {
            clusters_data.emplace_back();
            clusters_data.back().weight = weights(c);
            clusters_data.back().distribution = cluster_factory.makeRandomModel();
        }
        return std::make_unique<GaussianMixtureModel>(clusters_data);
    }
}
