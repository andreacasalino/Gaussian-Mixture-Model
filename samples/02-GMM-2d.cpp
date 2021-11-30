/**
 * Author:    Andrea Casalino
 * Created:   03.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "Utils.h"
#include <GaussianMixtureModel/ExpectationMaximization.h>
#include <GaussianMixtureModel/GaussianMixtureModel.h>
#include <GaussianMixtureModel/GaussianMixtureModelFactory.h>

Eigen::VectorXd make_2d_vector(const double val1, const double val2) {
  Eigen::VectorXd result(2);
  result << val1, val2;
  return result;
};

Eigen::MatrixXd make_2d_matrix(const Eigen::VectorXd &row1,
                               const Eigen::VectorXd &row2) {
  Eigen::MatrixXd result(2, 2);
  result.row(0) = row1;
  result.row(1) = row2;
  return result;
};

/////////////////////////
// GMM in 2 dimensions //
/////////////////////////
int main() {
  // build a reference model
  std::vector<gauss::gmm::Cluster> reference_clusters;
  {
    reference_clusters.emplace_back();
    reference_clusters.back().weight = 0.322522;
    reference_clusters.back().distribution =
        std::make_unique<gauss::GaussianDistribution>(
            make_2d_vector(0.24063, -0.604198),
            make_2d_matrix(make_2d_vector(0.186778, -0.00532367),
                           make_2d_vector(-0.00532367, 0.0516808)));
    reference_clusters.emplace_back();
    reference_clusters.back().weight = 0.336682;
    reference_clusters.back().distribution =
        std::make_unique<gauss::GaussianDistribution>(
            make_2d_vector(0.119904, 0.386934),
            make_2d_matrix(make_2d_vector(0.147499, -0.0454478),
                           make_2d_vector(-0.0454478, 0.128372)));
    reference_clusters.emplace_back();
    reference_clusters.back().weight = 0.214171;
    reference_clusters.back().distribution =
        std::make_unique<gauss::GaussianDistribution>(
            make_2d_vector(-0.666836, 0.0844356),
            make_2d_matrix(make_2d_vector(0.0342851, -0.0109963),
                           make_2d_vector(-0.0109963, 0.223311)));
    reference_clusters.emplace_back();
    reference_clusters.back().weight = 0.0209384;
    reference_clusters.back().distribution =
        std::make_unique<gauss::GaussianDistribution>(
            make_2d_vector(-0.851499, -0.837408),
            make_2d_matrix(make_2d_vector(0.00612019, 0.00756333),
                           make_2d_vector(0.00756333, 0.01204)));
    reference_clusters.emplace_back();
    reference_clusters.back().weight = 0.105686;
    reference_clusters.back().distribution =
        std::make_unique<gauss::GaussianDistribution>(
            make_2d_vector(0.739444, 0.577789),
            make_2d_matrix(make_2d_vector(0.01963, 0.00314812),
                           make_2d_vector(0.00314812, 0.0590969)));
  }
  gauss::gmm::GaussianMixtureModel reference_model(reference_clusters);

  // get samples from the reference model
  gauss::TrainSet train_set(reference_model.drawSamples(500));

  // fit a model using expectation maximization, considering the sampled train
  // set
  gauss::gmm::GaussianMixtureModel learnt_model(
      gauss::gmm::ExpectationMaximization(
          train_set, reference_model.getClusters().size()));

  // log the two models to visually check the differences
  print(reference_model, "reference_model2d.json");
  print(learnt_model, "learnt_model2d.json");

  // use the python script Visualize02.py to see the results

  return EXIT_SUCCESS;
}
