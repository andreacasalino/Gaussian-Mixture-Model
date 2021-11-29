/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "EvaluateLogDensity.h"
#include <GaussianMixtureModel/ExpectationMaximization.h>
#include <GaussianMixtureModel/KMeans.h>
#include <GaussianUtils/Utils.h>
#include <algorithm>
#include <limits>

namespace gauss::gmm {
namespace {
std::unique_ptr<std::vector<std::list<const Eigen::VectorXd *>>>
validateInitialLabeling(const std::size_t &N_clusters,
                        const std::vector<Eigen::VectorXd> &Samples,
                        const TrainInfo &info) {
  if (info.initialLabeling.empty())
    return nullptr;
  if (info.initialLabeling.size() != Samples.size())
    throw Error("Inconsistent number of labels for the passed initial guess");
  std::unique_ptr<std::vector<std::list<const Eigen::VectorXd *>>> clusters =
      std::make_unique<std::vector<std::list<const Eigen::VectorXd *>>>();
  std::size_t S = 0;
  std::for_each(info.initialLabeling.begin(), info.initialLabeling.end(),
                [&S](const std::size_t &l) {
                  if (l > S)
                    S = l;
                });
  if (N_clusters != (S + 1))
    throw Error("Inconsistent number of clusters for the passed initial guess");
  clusters->reserve(S + 1);
  for (std::size_t k = 0; k < S; ++k)
    clusters->emplace_back();
  auto itS = Samples.begin();
  std::for_each(info.initialLabeling.begin(), info.initialLabeling.end(),
                [&itS, &clusters](const std::size_t &l) {
                  (*clusters)[l].push_back(&(*itS));
                  ++itS;
                });
  for (std::size_t k = 0; k < S; ++k) {
    if ((*clusters)[k].empty())
      throw Error("Found empty cluster for the passed initial guess");
  };
  return clusters;
}

struct ClusterData {
  double weight;
  std::unique_ptr<GaussianDistribution> distribution;
};

std::vector<Cluster> convert(const std::vector<ClusterData> &clusters) {
  std::vector<Cluster> result;
  result.reserve(clusters.size());
  for (const auto &cluster : clusters) {
      result.emplace_back();
      result.back().weight = cluster.weight;
      result.back().distribution = std::make_unique<GaussianDistribution>(*cluster.distribution);
  }
  return result;
}
} // namespace

std::vector<Cluster> ExpectationMaximization(const TrainSet &train_set,
                                             const std::size_t &N_clusters,
                                             const TrainInfo &info,
                                             double *likelihood) {
  if (0 == N_clusters) {
    throw Error("Invalid number of clusters");
  }
  const auto &Samples = train_set.GetSamples();

  auto initialGuess = validateInitialLabeling(N_clusters, Samples, info);
  std::vector<std::list<const Eigen::VectorXd *>> clst;
  if (nullptr == initialGuess) {
    // use K means as initial guess
    kMeansClustering(clst, train_set, N_clusters, info.maxIterations);
  } else {
    clst = std::move(*initialGuess);
    initialGuess.reset();
  }

  std::vector<ClusterData> clusters;
  clusters.reserve(N_clusters);
  for (std::size_t k = 0; k < N_clusters; ++k) {
    Eigen::VectorXd Mean;
    Eigen::MatrixXd Cov;
    Cov = computeCovariance(
        clst[k], Mean, [](const Eigen::VectorXd *sample) { return *sample; });
    clusters.emplace_back();
    clusters.back().weight = 1.0 / static_cast<double>(clst.size());
    clusters.back().distribution =
        std::make_unique<GaussianDistribution>(Mean, Cov);
  }
  // EM loop
  std::size_t Iter = info.maxIterations;
  if (Iter < N_clusters)
    Iter = N_clusters;
  int R = static_cast<int>(Samples.size());
  int C = static_cast<int>(clusters.size());
  auto it_c = clusters.begin();
  Eigen::MatrixXd gamma(R, C);
  int r, c;
  double temp;
  Eigen::VectorXd n(C);
  double old_lkl = std::numeric_limits<double>::max(), new_lkl;
  std::vector<Eigen::VectorXd>::const_iterator it_s;
  for (int k = 0; k < Iter; ++k) {
    it_c = clusters.begin();
    for (c = 0; c < C; ++c) {
      it_s = Samples.begin();
      for (r = 0; r < R; ++r) {
        temp = it_c->distribution->evaluateLogDensity(*it_s);
        gamma(r, c) = it_c->weight * exp(temp);
        ++it_s;
      }
      ++it_c;
    }

    for (r = 0; r < R; ++r) {
      gamma.row(r) = (1.f / gamma.row(r).sum()) * gamma.row(r);
    }

    for (c = 0; c < C; ++c) {
      n(c) = gamma.col(c).sum();
    }

    c = 0;
    for (it_c = clusters.begin(); it_c != clusters.end(); ++it_c) {
      it_c->weight = n(c) / static_cast<double>(R);

      Eigen::VectorXd Mean(Samples.front().size());
      Mean.setZero();
      r = 0;
      for (it_s = Samples.begin(); it_s != Samples.end(); ++it_s) {
        Mean += gamma(r, c) * *it_s;
        ++r;
      }
      Mean *= 1.f / n(c);

      Eigen::MatrixXd Covariance(Samples.front().size(),
                                 Samples.front().size());
      Covariance.setZero();
      r = 0;
      for (it_s = Samples.begin(); it_s != Samples.end(); ++it_s) {
        Covariance += gamma(r, c) * (*it_s - Mean) * (*it_s - Mean).transpose();
        ++r;
      }
      Covariance *= 1.f / n(c);
      ++c;

      it_c->distribution =
          std::make_unique<GaussianDistribution>(Mean, Covariance);
    }

    new_lkl = 0.f;
    {
      std::vector<double> weights;
      weights.reserve(clusters.size());
      std::vector<const GaussianDistribution *> distributions;
      distributions.reserve(clusters.size());
      for (it_c = clusters.begin(); it_c != clusters.end(); ++it_c) {
        weights.push_back(it_c->weight);
        distributions.push_back(it_c->distribution.get());
      }
      for (it_s = Samples.begin(); it_s != Samples.end(); ++it_s) {
        new_lkl += evaluate_log_density(*it_s, weights, distributions);
      }
    }

    if (old_lkl != std::numeric_limits<double>::max()) {
      if (abs(old_lkl - new_lkl) < 1e-3) {
        old_lkl = new_lkl;
        break;
      }
    }
    old_lkl = new_lkl;
  }
  if (nullptr != likelihood) {
    *likelihood = old_lkl;
  }
  return convert(clusters);
}
} // namespace gauss::gmm