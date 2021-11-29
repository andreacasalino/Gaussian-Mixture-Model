/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianMixtureModel/GaussianMixtureModelSampler.h>

namespace gauss::gmm {
GaussianMixtureModelSampler::GaussianMixtureModelSampler(
    const GaussianMixtureModel &distribution) {
  const auto &clusters = distribution.getClusters();
  clusters_weights.reserve(clusters.size());
  clusters_samplers.reserve(clusters.size());
  for (const auto &cluster : clusters) {
    clusters_weights.push_back(cluster.weight);
    clusters_samplers.emplace_back(*cluster.distribution);
  }
}

Eigen::VectorXd GaussianMixtureModelSampler::getSample() const {
  std::size_t sampled_cluster = clusters_weights.size() - 1;
  {
    // sample the cluster index
    double r = unif_iso(generator);
    double cumulated = 0.0;
    for (std::size_t k = 0; k < clusters_weights.size(); ++k) {
      cumulated += clusters_weights[k];
      if (r < cumulated) {
        sampled_cluster = k;
        break;
      }
    }
  }
  // sample from the sampled cluster
  return clusters_samplers[sampled_cluster].getSample();
}
} // namespace gauss::gmm
