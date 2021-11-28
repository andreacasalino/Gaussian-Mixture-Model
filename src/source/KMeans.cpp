/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianMixtureModel/KMeans.h>
#include <algorithm>
#include <GaussianUtils/Utils.h>

namespace gauss::gmm {
	namespace {
		std::vector<Eigen::VectorXd> Forgy_init(const std::vector<Eigen::VectorXd>& samplesList, const std::size_t& N_means) {
			std::list<const Eigen::VectorXd*> samples;
			for (const auto& sample : samplesList) {
				samples.push_back(&sample);
			}

			std::vector< Eigen::VectorXd> Means;
			Means.reserve(N_means);
			std::size_t pos_rand;
			auto it_s = samples.begin();
			for (std::size_t k = 0; k < N_means; ++k) {
				pos_rand = rand() % samples.size();
				it_s = samples.begin();
				advance(it_s, pos_rand);
				Means.emplace_back(**it_s);
				it_s = samples.erase(it_s);
			}
			return Means;
		};
	}

	void kMeansClustering(std::vector<std::list<const Eigen::VectorXd*>>& clusters, const TrainSet& Samples, const std::size_t& N_cluster, const std::size_t& Iterations) {
		const auto& samplesList = Samples.GetSamples();

		if (0 == N_cluster) {
			throw Error("Invalid number of clusters");
		}
		if (N_cluster > samplesList.size()) {
			throw Error("Invalid number of clusters");
		}

		clusters.clear();
		for (std::size_t k = 0; k < N_cluster; ++k) clusters.push_back({});
		auto Means = Forgy_init(samplesList, N_cluster);

		std::size_t Iter = Iterations;
		if (Iter < N_cluster) {
			Iter = N_cluster;
		}

		double dist_min, temp;
		std::size_t pos_nearest, kk;
		std::vector<std::list<const Eigen::VectorXd*>> old_clustering;
		for (std::size_t k = 0; k < Iter; ++k) {
			// recompute clusters
			for (kk = 0; kk < clusters.size(); ++kk ) {
				clusters[kk].clear();
			}
			std::for_each(samplesList.begin(), samplesList.end(), [&](const Eigen::VectorXd& v) {
				pos_nearest = 0;
				dist_min = (v - Means[0]).squaredNorm();
				for (kk = 1; kk < N_cluster; ++kk) {
					temp = (v - Means[kk]).squaredNorm();
					if (temp < dist_min) {
						dist_min = temp;
						pos_nearest = kk;
					}
				}
				clusters[pos_nearest].push_back(&v);
			});
			// recompute means
			for (kk = 0; kk < N_cluster; ++kk) {
				Means[kk] = computeMean(clusters[kk], [](const auto* sample) {return *sample; });
			}
			// check clustering changed w.r.t to previous iteration
			if (!old_clustering.empty() && (old_clustering == clusters)) {
					return;
			}
			old_clustering = clusters;
		}
	}
}