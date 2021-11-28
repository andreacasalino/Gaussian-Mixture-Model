#include <Packer.h>
#include <gtest/gtest.h>

#include <GaussianMixtureModel/GaussianMixtureModel.h>

TEST(DistributionCreation, postive_tests) {
    {
        std::vector<gauss::gmm::Cluster> clusters;
        clusters.reserve(2);
        clusters.emplace_back(1, gauss::GaussianDistribution(gauss::test::make_vector({ 1, -1 }),
            gauss::test::make_matrix({ {1, 0}, {0, 2} })));
        clusters.emplace_back(2, gauss::GaussianDistribution(gauss::test::make_vector({ -1, 1 }),
            gauss::test::make_matrix({ {2, 0}, {0, 1} })));
        EXPECT_NO_THROW(gauss::gmm::GaussianMixtureModel{ clusters };);
    }

    {
        std::vector<gauss::gmm::Cluster> clusters;
        clusters.reserve(3);
        clusters.emplace_back(1, gauss::GaussianDistribution(gauss::test::make_vector({ 1, -1 }),
            gauss::test::make_matrix({ {1, 0}, {0, 2} })));
        clusters.emplace_back(3, gauss::GaussianDistribution(gauss::test::make_vector({ 0, 0 }),
            gauss::test::make_matrix({ {1, 0}, {0, 1} })));
        clusters.emplace_back(2, gauss::GaussianDistribution(gauss::test::make_vector({ -1, 1 }),
            gauss::test::make_matrix({ {2, 0}, {0, 1} })));
        EXPECT_NO_THROW(gauss::gmm::GaussianMixtureModel{ clusters });
    }
}

TEST(DistributionCreation, negative_tests) {
    {
        std::vector<gauss::gmm::Cluster> clusters;
        EXPECT_THROW(gauss::gmm::GaussianMixtureModel{ clusters }, gauss::gmm::Error);
    }

    {
        std::vector<gauss::gmm::Cluster> clusters;
        clusters.reserve(3);
        clusters.emplace_back(1, gauss::GaussianDistribution(gauss::test::make_vector({ 1, -1 }),
            gauss::test::make_matrix({ {1, 0}, {0, 2} })));
        clusters.emplace_back(-3, gauss::GaussianDistribution(gauss::test::make_vector({ 0, 0 }),
            gauss::test::make_matrix({ {1, 0}, {0, 1} })));
        clusters.emplace_back(-2, gauss::GaussianDistribution(gauss::test::make_vector({ -1, 1 }),
            gauss::test::make_matrix({ {2, 0}, {0, 1} })));
        EXPECT_THROW(gauss::gmm::GaussianMixtureModel{ clusters }, gauss::gmm::Error);
    }

    {
        std::vector<gauss::gmm::Cluster> clusters;
        clusters.reserve(3);
        clusters.emplace_back(1, gauss::GaussianDistribution(gauss::test::make_vector({ 1, -1 }),
            gauss::test::make_matrix({ {1, 0}, {0, 2} })));
        clusters.emplace_back(3, gauss::GaussianDistribution(gauss::test::make_vector({ 0, 0, 0 }),
            gauss::test::make_matrix({ {1, 0, 0}, {0, 1, 0}, {0, 0, 1} })));
        clusters.emplace_back(2, gauss::GaussianDistribution(gauss::test::make_vector({ -1, 1 }),
            gauss::test::make_matrix({ {2, 0}, {0, 1} })));
        EXPECT_THROW(gauss::gmm::GaussianMixtureModel{ clusters }, gauss::gmm::Error);
    }
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
