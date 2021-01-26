/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <header/GMM.h>
#include <header/KMeans.h>
#include "Commons.h"
#include <memory>
#include <algorithm>
#include <limits>

namespace gmm {
	std::unique_ptr<std::vector<std::list<const V*>>> validateInitialLabeling(const std::size_t& N_clusters, const std::list<V>& Samples, const GMM::TrainInfo& info) {
		if (info.initialLabeling.empty()) return nullptr;
		if (info.initialLabeling.size() != Samples.size()) throw Error("Inconsistent number of labels for the passed initial guess");
		std::unique_ptr<std::vector<std::list<const V*>>> clusters = std::make_unique<std::vector<std::list<const V*>>>();
		std::size_t S = 0;
		std::for_each(info.initialLabeling.begin(), info.initialLabeling.end(), [&S](const std::size_t& l) {
			if (l > S) S = l;
		});
		if (N_clusters != (S + 1)) throw Error("Inconsistent number of clusters for the passed initial guess");
		clusters->reserve(S + 1);
		for (std::size_t k = 0; k < S; ++k) clusters->emplace_back();
		auto itS = Samples.begin();
		std::for_each(info.initialLabeling.begin(), info.initialLabeling.end(), [&itS, &clusters](const std::size_t& l) {
			(*clusters)[l].push_back(&(*itS));
			++itS;
		});
		for (std::size_t k = 0; k < S; ++k) {
			if ((*clusters)[k].empty()) throw Error("Found empty cluster for the passed initial guess");
		};
		return clusters;
	}

	double GMM::ExpectationMaximization(const TrainSet& train_set, const std::size_t& N_clusters, const TrainInfo& info) {
		if (0 == N_clusters) throw Error("Invalid number of clusters");
		const std::list<V>& Samples = train_set.GetSamples();

		auto initialGuess = validateInitialLabeling(N_clusters, Samples, info);
		std::vector<std::list<const V*>> clst;
		if (nullptr == initialGuess) {
			// use K means as initial guess
			kMeansClustering(clst, train_set, N_clusters, info.maxIterations);
		}
		else {
			clst = std::move(*initialGuess);
			initialGuess.reset();
		}

		V Mean;
		M Cov;
		this->Clusters.reserve(N_clusters);
		for (std::size_t k = 0; k < N_clusters; ++k) {
			Mean = mean(clst[k]);
			Cov = covariance(clst[k], Mean);
			this->appendCluster(1.0 / static_cast<double>(clst.size()) , Mean, Cov);
		}
		//EM loop
		std::size_t Iter = info.maxIterations;
		if (Iter < N_clusters) Iter = N_clusters;
		int R = static_cast<int>(Samples.size());
		int C = static_cast<int>(this->Clusters.size());
		auto it_c = this->Clusters.begin();
		M gamma(R, C);
		int r, c;
		double temp;
		V n(C);
		double old_lkl = std::numeric_limits<double>::max(), new_lkl;
		std::list<V>::const_iterator it_s;
		for (int k = 0; k < Iter; ++k) {
			it_c = this->Clusters.begin();
			for (c = 0; c < C; ++c) {
				it_s = Samples.begin();
				for (r = 0; r < R; ++r) {
					temp = evalNormalLogDensity(*it_c, *it_s);
					gamma(r, c) = it_c->weight * exp(temp);
					++it_s;
				}
				++it_c;
			}

			for (r = 0; r < R; ++r)
				gamma.row(r) = (1.f / gamma.row(r).sum()) * gamma.row(r);

			for (c = 0; c < C; ++c)
				n(c) = gamma.col(c).sum();

			c = 0;
			for (it_c = this->Clusters.begin(); it_c != this->Clusters.end(); ++it_c) {
				it_c->weight = n(c) / static_cast<double>(R);

				it_c->Mean.setZero();
				r = 0;
				for (it_s = Samples.begin(); it_s != Samples.end(); ++it_s) {
					it_c->Mean += gamma(r, c) * *it_s;
					++r;
				}
				it_c->Mean *= 1.f / n(c);

				it_c->Covariance.setZero();
				r = 0;
				for (it_s = Samples.begin(); it_s != Samples.end(); ++it_s) {
					it_c->Covariance += gamma(r, c) * (*it_s - it_c->Mean) * (*it_s - it_c->Mean).transpose();
					++r;
				}
				it_c->Covariance *= 1.f / n(c);
				++c;

				invertSymmPositive(it_c->Inverse_Cov, it_c->Covariance);
				it_c->Abs_Deter_Cov = abs(it_c->Covariance.determinant());
			}

			new_lkl = 0.f;
			for (it_s = Samples.begin(); it_s != Samples.end(); ++it_s) {
				new_lkl += this->getLogDensity(*it_s);
			}

			if (old_lkl != std::numeric_limits<double>::max()) {
				if (abs(old_lkl - new_lkl) < 1e-3)
					return new_lkl;
			}
			old_lkl = new_lkl;
		}
		return old_lkl;
	}
}