/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <header/GMM.h>
#include <math.h>
#include "Commons.h"
using namespace std;

namespace gmm {
	GMM::GMM(const size_t& N_clusters, const TrainSet& train_set, const TrainInfo& info) {
		this->ExpectationMaximization(train_set, N_clusters, info);
	}

	GMM::GMM(const size_t& N_clusters, const size_t& dimension_size) {
		if (0 == N_clusters) throw Error("Invalid number of clusters");
		if (0 == dimension_size) throw Error("Invalid space size");

		size_t N_sample = 100 * N_clusters;

		list<V> samples;
		size_t kk;
		for (size_t k = 0; k < N_sample; ++k) {
			samples.emplace_back(dimension_size);
			for (kk = 0; kk < dimension_size; ++kk)
				samples.back()(kk) = 2.0 * static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - 1.0;
		}
		this->ExpectationMaximization(TrainSet(samples), N_clusters, TrainInfo{});
	}

	GMM::GMM(const M& params) {
		size_t x = (size_t)params.cols();
		size_t Ncl = (size_t)params.rows() / (x + 2);
		if (Ncl * (x + 2) != params.rows()) throw Error("Invalid GMM parameters");

		size_t r;
		V Mean(x);
		M Cov(x, x);
		double w;
		for (size_t k = 0; k < Ncl; ++k) {
			w = params((2 + x) * k, 0);
			if (w < 0) throw Error("Invalid GMM parameters");
			if (w > 1.f) throw Error("Invalid GMM parameters");

			Mean = params.row((2 + x) * k + 1).transpose();
			for (r = 0; r < x; r++)
				Cov.row(r) = params.row((2 + x) * k + r + 2);
			if(!checkCovariance(Cov)) Error("Invalid GMM parameters");

			this->appendCluster(w, Mean, Cov);
		}
	}

	GMM GMM::fitOptimalModel(const TrainSet& train_set, const std::vector<size_t>& N_clusters_to_try, const size_t& Iterations) {
		if (N_clusters_to_try.empty()) throw Error("invalid clusters sizes to try");

		struct model {
			GMM* model;
			double lkl;
		};
		list<model> models;
		for (auto it = N_clusters_to_try.begin(); it != N_clusters_to_try.end(); it++) {
			models.push_back(model());
			TrainInfo info;
			info.maxIterations = Iterations;
			models.back().model = new GMM(*it, train_set, info);
			models.back().lkl = models.back().model->getLogLikelihood(train_set);
		}

		auto it = models.begin();
		model* best_model = &(*it);
		it++;
		for (it; it != models.end(); it++) {
			if (it->lkl > best_model->lkl)
				best_model = &(*it);
		}

		GMM temp(*best_model->model);

		for (it = models.begin(); it != models.end(); it++) delete it->model;

		return temp;
	}

	void GMM::appendCluster(const double& w, const V& mean, const M& cov) {
		this->Clusters.emplace_back();
		this->Clusters.back().weight = w;
		this->Clusters.back().Mean = mean;
		this->Clusters.back().Covariance = cov;
		invertSymmPositive(this->Clusters.back().Inverse_Cov, this->Clusters.back().Covariance);
		this->Clusters.back().Abs_Deter_Cov = abs(this->Clusters.back().Covariance.determinant());
	}

	double GMM::getLogDensity(const V& X) const {
		double den = 0.f;
		std::for_each(this->Clusters.begin(), this->Clusters.end(), [&den, &X](const GMMcluster& cl) {
			den += exp(log(cl.weight) + evalNormalLogDensity(cl, X));
		});
		den = log(den);
		return den;
	};

	double GMM::getLogLikelihood(const TrainSet& train_set) {
		double lik = 0.0;
		std::for_each(train_set.GetSamples().begin(), train_set.GetSamples().end(), [this, &lik](const V& s) {
			lik += this->getLogDensity(s);
		});
		return lik;
	}

	V GMM::Classify(const V& X) const {
		V cls(this->Clusters.size());
		for (std::size_t k = 0; k < this->Clusters.size(); ++k) {
			cls(k) = exp(log(this->Clusters[k].weight) + evalNormalLogDensity(this->Clusters[k], X));
		}
		cls *= (1.0 / cls.sum());
		return cls;
	}

	M GMM::getMatrixParameters() const {
		size_t D = 2 + this->Clusters.front().Mean.size();
		M Matr(this->Clusters.size() * D, this->Clusters.front().Mean.size());
		size_t r, R = this->Clusters.front().Mean.size();
		for (size_t k = 0; k < this->Clusters.size(); k++) {
			Matr.row(D * k).setZero();
			Matr.row(D * k)(0) = this->Clusters[k].weight;
			Matr.row(D * k + 1) = this->Clusters[k].Mean.transpose();

			for (r = 0; r < R; r++)
				Matr.row(D * k + r + 2) = this->Clusters[k].Covariance.row(r);
		}
		return Matr;
	}
}
