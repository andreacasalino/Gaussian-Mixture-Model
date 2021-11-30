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

Eigen::VectorXd make_3d_vector(const double val1, const double val2,
                               const double val3) {
  Eigen::VectorXd result(3);
  result << val1, val2, val3;
  return result;
};

Eigen::MatrixXd make_3d_matrix(const Eigen::VectorXd &row1,
                               const Eigen::VectorXd &row2,
                               const Eigen::VectorXd &row3) {
  Eigen::MatrixXd result(3, 3);
  result.row(0) = row1;
  result.row(1) = row2;
  result.row(2) = row3;
  return result;
};

/////////////////////////
// GMM in 3 dimensions //
/////////////////////////
int main() {
  // build a reference model
  std::vector<gauss::gmm::Cluster> reference_clusters;
  {
    reference_clusters.emplace_back();
    reference_clusters.back().weight = 0.146031;
    reference_clusters.back().distribution =
        std::make_unique<gauss::GaussianDistribution>(
            make_3d_vector(0.443929, 0.139176, -0.68346),
            make_3d_matrix(make_3d_vector(0.0869121, 0.0291794, -0.00301252),
                           make_3d_vector(0.0291794, 0.286838, 0.0161968),
                           make_3d_vector(-0.00301252, 0.0161968, 0.0379466)));
    reference_clusters.emplace_back();
    reference_clusters.back().weight = 0.224085;
    reference_clusters.back().distribution =
        std::make_unique<gauss::GaussianDistribution>(
            make_3d_vector(-0.144286, -0.589395, 0.0368512),
            make_3d_matrix(make_3d_vector(0.285022, 0.00539453, -0.0686665),
                           make_3d_vector(0.00539453, 0.0585836, -0.00108196),
                           make_3d_vector(-0.0686665, -0.00108196, 0.192618)));
    reference_clusters.emplace_back();
    reference_clusters.back().weight = 0.134425;
    reference_clusters.back().distribution =
        std::make_unique<gauss::GaussianDistribution>(
            make_3d_vector(0.542868, -0.211808, 0.318758),
            make_3d_matrix(make_3d_vector(0.0684148, 5.54491e-05, 0.0099768),
                           make_3d_vector(5.54491e-05, 0.198069, -0.0977015),
                           make_3d_vector(0.0099768, -0.0977015, 0.11236)));
    reference_clusters.emplace_back();
    reference_clusters.back().weight = 0.224586;
    reference_clusters.back().distribution =
        std::make_unique<gauss::GaussianDistribution>(
            make_3d_vector(-0.131381, 0.507175, 0.174051),
            make_3d_matrix(make_3d_vector(0.245183, 0.00509693, 0.0926525),
                           make_3d_vector(0.00509693, 0.082708, -0.0021128),
                           make_3d_vector(0.0926525, -0.0021128, 0.15217)));
    reference_clusters.emplace_back();
    reference_clusters.back().weight = 0.102773;
    reference_clusters.back().distribution =
        std::make_unique<gauss::GaussianDistribution>(
            make_3d_vector(-0.619147, 0.00883687, -0.730442),
            make_3d_matrix(make_3d_vector(0.0483469, 0.0401095, -0.00529763),
                           make_3d_vector(0.0401095, 0.359586, -0.0315178),
                           make_3d_vector(-0.00529763, -0.0315178, 0.0282302)));
    reference_clusters.emplace_back();
    reference_clusters.back().weight = 0.1681;
    reference_clusters.back().distribution =
        std::make_unique<gauss::GaussianDistribution>(
            make_3d_vector(-0.110604, 0.0216821, 0.794906),
            make_3d_matrix(make_3d_vector(0.266708, 0.0294118, 0.0174374),
                           make_3d_vector(0.0294118, 0.30424, 0.003477),
                           make_3d_vector(0.0174374, 0.003477, 0.0141536)));
  }
  gauss::gmm::GaussianMixtureModel reference_model(reference_clusters);

  // get samples from the reference model
  gauss::TrainSet train_set(reference_model.drawSamples(1000));

  // fit a model using expectation maximization, considering the sampled train
  // set
  gauss::gmm::GaussianMixtureModel learnt_model(
      gauss::gmm::ExpectationMaximization(
          train_set, reference_model.getClusters().size()));

  // log the two models to visually check the differences
  print(reference_model, "reference_model3d.json");
  print(learnt_model, "learnt_model3d.json");

  // use the python script Visualize03.py to see the results

  return EXIT_SUCCESS;
}
