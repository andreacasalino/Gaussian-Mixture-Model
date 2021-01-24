/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "Commons.h"
#include <algorithm>
#include <Eigen/Cholesky>

namespace gmm {
	V mean(const std::list<const V*>& l) {
		if (l.empty()) throw Error("cannot compute mean of empty collection");
		auto it = l.begin();
		V Mean = *l.front();
		++it;
		std::for_each(it, l.end(), [&Mean](const V* pt) {
			Mean += *pt;
		});
		Mean = (1.0 / static_cast<double>(l.size()) ) * Mean;
		return Mean;
	}

	M covariance(const std::list<const V*>& l, const V& Mean) {
		if (l.empty()) throw Error("cannot compute covariance of empty collection");
		M Cov(Mean.size(), Mean.size());
		Cov.setZero();
		std::for_each(l.begin(), l.end(), [&Cov, &Mean](const V* pt) {
			Cov += (*pt - Mean) * (*pt - Mean).transpose();
		});
		Cov = (1.0 / static_cast<double>(l.size()))* Cov;
		return Cov;
	}

	void invertSymmPositive(M& Sigma_inverse, const M& Sigma) {
		//LLT<MatrixXf> lltOfCov(Sigma);
		//MatrixXf L(lltOfCov.matrixL());
		//*Sigma_inverse = L * L.transpose();

		Sigma_inverse = Sigma.inverse();
	}

	double evalNormalLogDensity(const GMMcluster& distr, const V& X) {
		double den;
		den = (X - distr.Mean).transpose() * distr.Inverse_Cov * (X - distr.Mean);
		den += X.size() * LOG_2_PI;
		den += log(distr.Abs_Deter_Cov);
		den *= -0.5;
		return den;
	}

	bool checkCovariance(const M& Cov) {
		std::size_t K = (std::size_t)Cov.cols();
		std::size_t c;
		for (std::size_t r = 0; r < K; ++r) {
			for (c = (r + 1); c < K; ++c) {
				if (abs(Cov(r, c) - Cov(c, r)) > 1e-5) return false;
			}
		}

		Eigen::EigenSolver<M> eig_solv(Cov);
		auto eigs = eig_solv.eigenvalues();

		//check all values are sufficient high
		for (std::size_t k = 0; k < K; ++k) {
			if (abs(eigs(k).real()) < 1e-5) return false;
		}
		return true;
	}
}