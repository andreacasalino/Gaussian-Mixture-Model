#include <GaussianMixtureModel/GaussianMixtureModelFactory.h>
#include <GaussianMixtureModel/ExpectationMaximization.h>
#include <Packer.h>
#include <gtest/gtest.h>

constexpr std::size_t SAMPLES = 100000;

constexpr double TOLL = 0.025;
#define EXPECT_SIMILAR(VAL_A, VAL_B) EXPECT_LE(abs(VAL_A - VAL_B), TOLL)

void expect_similar(const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
  EXPECT_EQ(a.size(), b.size());
  for (std::size_t k = 0; k < a.size(); ++k) {
    EXPECT_SIMILAR(a(k), b(k));
  }
};

void expect_similar(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
  EXPECT_EQ(a.rows(), b.rows());
  EXPECT_EQ(a.cols(), b.cols());
  for (std::size_t c = 0; c < a.cols(); ++c) {
    for (std::size_t r = 0; r < a.rows(); ++r) {
      EXPECT_SIMILAR(a(r, c), b(r, c));
    }
  }
};

void expect_similar(const std::vector<gauss::gmm::Cluster>& a, const std::vector<gauss::gmm::Cluster>& b) {
    EXPECT_EQ(a.size(), b.size());
    for (std::size_t k = 0; k < a.size(); ++k) {
        EXPECT_SIMILAR(a[k].weight, b[k].weight);
        expect_similar(a[k].distribution.getMean(), b[k].distribution.getMean());
        expect_similar(a[k].distribution.getCovariance(), b[k].distribution.getCovariance());
    }
}

TEST(Sampling, 3d) {
    std::vector<gauss::gmm::Cluster> clusters;
    {        
        clusters.reserve(3);
        clusters.emplace_back(1, gauss::GaussianDistribution(gauss::test::make_vector({ 1, -1 }),
            gauss::test::make_matrix({ {0.1, 0}, {0, 0.2} })));
        clusters.emplace_back(3, gauss::GaussianDistribution(gauss::test::make_vector({ 0, 0 }),
            gauss::test::make_matrix({ {0.3, 0}, {0, 0.3} })));
        clusters.emplace_back(2, gauss::GaussianDistribution(gauss::test::make_vector({ -1, 1 }),
            gauss::test::make_matrix({ {0.2, 0}, {0, 0.1} })));
    }

  auto samples = gauss::gmm::GaussianMixtureModel(clusters).drawSamples(SAMPLES);

  auto learnt_clusters = gauss::gmm::ExpectationMaximization(samples, clusters.size());
  expect_similar(clusters, learnt_clusters);

  gauss::gmm::GaussianMixtureModel learnt_model(learnt_clusters);
  expect_similar(clusters, learnt_model.getClusters());
}

TEST(Sampling, 6d) {
    auto sampled_model = gauss::gmm::GaussianMixtureModelFactory(6, 2).makeRandomModel();

    auto samples = sampled_model->drawSamples(SAMPLES);

    auto learnt_clusters = gauss::gmm::ExpectationMaximization(samples, sampled_model->getClusters().size());
    expect_similar(sampled_model->getClusters(), learnt_clusters);

    gauss::gmm::GaussianMixtureModel learnt_model(learnt_clusters);
    expect_similar(sampled_model->getClusters(), learnt_model.getClusters());
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
