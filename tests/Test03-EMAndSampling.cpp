#include <GaussianMixtureModel/ExpectationMaximization.h>
#include <GaussianMixtureModel/GaussianMixtureModelFactory.h>
#include <Packer.h>
#include <gtest/gtest.h>
#include <set>

constexpr std::size_t SAMPLES = 3000;

#define EXPECT_SIMILAR(VAL_A, VAL_B, TOLL) EXPECT_LE(abs(VAL_A - VAL_B), TOLL)

void expect_similar(const Eigen::VectorXd &mean_a,
                    const Eigen::VectorXd &mean_b) {
  EXPECT_EQ(mean_a.size(), mean_b.size());
  for (std::size_t k = 0; k < mean_a.size(); ++k) {
    EXPECT_SIMILAR(mean_a(k), mean_b(k), 0.05);
  }
};

void expect_similar(const Eigen::MatrixXd &cov_a,
                    const Eigen::MatrixXd &cov_b) {
  EXPECT_EQ(cov_a.rows(), cov_b.rows());
  EXPECT_EQ(cov_a.cols(), cov_b.cols());
  for (std::size_t c = 0; c < cov_a.cols(); ++c) {
    for (std::size_t r = 0; r < cov_a.rows(); ++r) {
      EXPECT_SIMILAR(cov_a(r, c), cov_b(r, c), 0.05);
    }
  }
};

void expect_similar(const gauss::gmm::Cluster &a,
                    const gauss::gmm::Cluster &b) {
  EXPECT_SIMILAR(a.weight, b.weight, 0.1);
  expect_similar(a.distribution->getMean(), b.distribution->getMean());
  expect_similar(a.distribution->getCovariance(),
                 b.distribution->getCovariance());
}

const gauss::gmm::Cluster *
get_closest(const gauss::gmm::Cluster &a,
            const std::set<const gauss::gmm::Cluster *> &remaining) {
  auto it = remaining.begin();
  const gauss::gmm::Cluster *closest = *it;
  double min_distance =
      Eigen::VectorXd(a.distribution->getMean() - (*it)->distribution->getMean())
          .norm();
  ++it;
  for (it; it != remaining.end(); ++it) {
    double att_distance = Eigen::VectorXd(a.distribution->getMean() -
                                          (*it)->distribution->getMean())
                              .norm();
    if (att_distance < min_distance) {
      closest = *it;
      min_distance = att_distance;
    }
  }
  return closest;
}
void expect_similar(const std::vector<gauss::gmm::Cluster> &a,
                    const std::vector<gauss::gmm::Cluster> &b) {
  EXPECT_EQ(a.size(), b.size());
  std::set<const gauss::gmm::Cluster *> remaining;
  for (const auto &b_cluster : b) {
    remaining.emplace(&b_cluster);
  }
  for (const auto &a_cluster : a) {
    const auto *closest = get_closest(a_cluster, remaining);
    expect_similar(*closest, a_cluster);
    remaining.erase(closest);
  }
}

void emplace_back(std::vector<gauss::gmm::Cluster>& clusters, const double w, const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance) {
    clusters.emplace_back();
    clusters.back().weight = w;
    clusters.back().distribution = std::make_unique<gauss::GaussianDistribution>(mean, covariance);
}

TEST(Sampling, 3d) {
  std::vector<gauss::gmm::Cluster> clusters;
  {
    Eigen::VectorXd eigs_cov(2);
    eigs_cov << 0.1, 0.2;
    clusters.reserve(3);

    emplace_back(clusters, 0.5, gauss::test::make_vector({0, 0}),
                                   gauss::make_random_covariance(eigs_cov));

    emplace_back(clusters, 0.25, gauss::test::make_vector({2, 2}),
                                    gauss::make_random_covariance(eigs_cov));

    emplace_back(clusters, 0.25, gauss::test::make_vector({-2, -2}),
                                    gauss::make_random_covariance(eigs_cov));
  }

  auto samples =
      gauss::gmm::GaussianMixtureModel(clusters).drawSamples(SAMPLES);

  auto learnt_clusters =
      gauss::gmm::ExpectationMaximization(samples, clusters.size());
  expect_similar(clusters, learnt_clusters);

  gauss::gmm::GaussianMixtureModel learnt_model(learnt_clusters);
  expect_similar(clusters, learnt_model.getClusters());
}

TEST(Sampling, 6d) {
  std::vector<gauss::gmm::Cluster> clusters;
  {
    Eigen::VectorXd eigs_cov(6);
    eigs_cov << 0.1, 0.2, 0.1, 0.2, 0.1, 0.2;
    clusters.reserve(2);

    emplace_back(clusters, 0.5, gauss::test::make_vector({3, 3, 3, 3, 3, 3}),
                                   gauss::make_random_covariance(eigs_cov));

    emplace_back(clusters, 0.5, gauss::test::make_vector({-3, -3, -3, -3, -3, -3}),
                                    gauss::make_random_covariance(eigs_cov));
  }

  gauss::gmm::GaussianMixtureModel ref_model(clusters);

  auto samples = ref_model.drawSamples(SAMPLES);

  auto learnt_clusters = gauss::gmm::ExpectationMaximization(
      samples, ref_model.getClusters().size());
  expect_similar(ref_model.getClusters(), learnt_clusters);

  gauss::gmm::GaussianMixtureModel learnt_model(learnt_clusters);
  expect_similar(ref_model.getClusters(), learnt_model.getClusters());
}

TEST(Sampling, 2d) {
  std::vector<gauss::gmm::Cluster> clusters;
  const std::size_t nClusters = 4;
  {
    Eigen::VectorXd eigs_cov(2);
    eigs_cov << 0.1, 0.2;
    clusters.reserve(nClusters);
    double angle = 0.0;
    double angle_delta = gauss::PI_GREEK / static_cast<double>(nClusters);
    for (std::size_t c = 0; c < nClusters; ++c) {
      emplace_back(clusters,
          1.0 / static_cast<double>(nClusters),
              gauss::test::make_vector({2.0 * cos(angle), 2.0 * sin(angle)}),
              gauss::make_random_covariance(eigs_cov));
      angle += angle_delta;
    }
  }

  gauss::gmm::GaussianMixtureModel ref_model(clusters);

  auto samples = ref_model.drawSamples(SAMPLES);

  auto learnt_clusters = gauss::gmm::ExpectationMaximization(
      samples, ref_model.getClusters().size());
  expect_similar(ref_model.getClusters(), learnt_clusters);

  gauss::gmm::GaussianMixtureModel learnt_model(learnt_clusters);
  expect_similar(ref_model.getClusters(), learnt_model.getClusters());
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
