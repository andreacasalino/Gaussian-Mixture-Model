#include <GaussianMixtureModel/KMeans.h>
#include <GaussianUtils/Utils.h>
#include <gtest/gtest.h>

constexpr double BIG_RAY = 20;
constexpr std::size_t SAMPLES_X_CLUSTER = 50;
gauss::TrainSet make_samples(const std::size_t clusters) {
    std::vector<Eigen::VectorXd> centers;
    centers.reserve(clusters);
    double angle = 0.0;
    double angle_delta = gauss::PI_GREEK * 2.0 / static_cast<double>(clusters);
    for (std::size_t c = 0; c < clusters; ++c) {
        centers.emplace_back(2);
        centers.back()(0) = BIG_RAY * cos(angle);
        centers.back()(1) = BIG_RAY * sin(angle);
        angle += angle_delta;
    }
    std::vector<Eigen::VectorXd> samples;
    samples.reserve(SAMPLES_X_CLUSTER * clusters);
    for (std::size_t s = 0; s < SAMPLES_X_CLUSTER; ++s) {
        for(std::size_t c = 0; c < clusters; ++c) {
            samples.emplace_back(2);
            samples.back().setRandom();
            samples.back() += centers[c];
        }
    }
    return gauss::TrainSet{ samples };
};

constexpr double SMALL_RAY = 2;
void check_classification(const std::vector<std::list<const Eigen::VectorXd*>>& clusters) {
    for (const auto& cluster : clusters) {
        auto mean = gauss::computeMean(cluster, [](const auto* sample) { return *sample; });
        for (const Eigen::VectorXd* sample : cluster) {
            EXPECT_LE(Eigen::VectorXd(*sample - mean).norm(), SMALL_RAY);
        }
    }
};

void make_samples_and_check_classification(const std::size_t clusters) {
    auto samples = make_samples(clusters);
    std::vector<std::list<const Eigen::VectorXd*>> result;
    gauss::gmm::kMeansClustering(result, samples, clusters);
    check_classification(result);
}

TEST(kMeans, three_clusters) {
    make_samples_and_check_classification(3);
}

TEST(kMeans, four_clusters) {
    make_samples_and_check_classification(4);
}

TEST(kMeans, five_clusters) {
    make_samples_and_check_classification(5);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
