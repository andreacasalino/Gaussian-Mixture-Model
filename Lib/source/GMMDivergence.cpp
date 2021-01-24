/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <header/GMM.h>
#include "Commons.h"

namespace gmm {
	double GMM::getKullbackLeiblerDiergenceMonteCarlo(const GMM& other) const {
		if (this->Clusters.front().Mean.size() != other.Clusters.front().Mean.size()) throw Error("The 2 GMM are not comparable");
		std::list<V> Samples;
		Samples = this->drawSamples((size_t)this->Clusters.front().Mean.size() * 500);
		double div = 0.f;
		for (auto it = Samples.begin(); it != Samples.end(); ++it) {
			div += this->getLogDensity(*it);
			div -= other.getLogDensity(*it);
		}
		div *= 1.0 / static_cast<double>(Samples.size());
		return div;
	}

	double divergenceNormals(const GMMcluster& f, const GMMcluster& g) {
		double temp = log(g.Abs_Deter_Cov) - log(f.Abs_Deter_Cov);
		M P = g.Inverse_Cov * f.Covariance;
		temp += P.trace();
		V Delta = f.Mean - g.Mean;
		temp += Delta.transpose() * g.Inverse_Cov * Delta;
		temp -= static_cast<double>(f.Mean.size());
		temp *= 0.5f;
		return temp;
	}

	double tOperator(const GMMcluster& f, const GMMcluster& g) {
		double temp = -f.Mean.size() * log(2.0 * PI_GREEK);
		M S = f.Covariance;
		S += g.Covariance;
		temp -= log(S.determinant());
		V Delta = g.Mean - f.Mean;
		temp -= Delta.transpose() * S.inverse() * Delta;
		temp *= 0.5;
		return exp(temp);
	}

	std::pair<double, double> GMM::getKullbackLeiblerDiergenceEstimate(const GMM& other) const {
		if (this->Clusters.front().Mean.size() != other.Clusters.front().Mean.size()) throw Error("The 2 GMM are not comparable");

		M Divergences(this->Clusters.size(), other.Clusters.size());
		M t(this->Clusters.size(), other.Clusters.size());
		M z(this->Clusters.size(), this->Clusters.size());
		size_t a, A = this->Clusters.size(), b, B = other.Clusters.size();
		for (a = 0; a < A; a++) {
			for (b = 0; b < B; b++) {
				Divergences(a, b) = divergenceNormals(this->Clusters[a], other.Clusters[b]);
				t(a, b) = tOperator(this->Clusters[a], other.Clusters[b]);
			}
		}
		for (a = 0; a < A; a++) {
			for (b = 0; b < A; b++) {
				z(a, b) = tOperator(this->Clusters[a], this->Clusters[b]);
			}
		}

		// <lower , upper> bound
		std::pair<double, double> bound = std::make_pair<double, double>(0.0, 0.0);
		size_t a2;
		double temp = 0.f, temp2, temp3;
		for (a = 0; a < A; a++) {
			temp2 = 0.f;
			for (a2 = 0; a2 < A; a2++)  temp2 += this->Clusters[a2].weight * z(a, a2);
			temp2 = log(temp2);
			temp3 = 0.f;
			for (b = 0; b < B; b++)  temp3 += other.Clusters[b].weight * exp(-Divergences(a, b));
			temp3 = log(temp3);
			bound.second += this->Clusters[a].weight * (temp2 - temp3);

			temp2 = 0.f;
			for (a2 = 0; a2 < A; a2++)  temp2 += this->Clusters[a2].weight * exp(-Divergences(a, b));
			temp2 = log(temp2);
			temp3 = 0.f;
			for (b = 0; b < B; b++)  temp3 += other.Clusters[b].weight * t(a, b);
			temp3 = log(temp3);
			bound.first += this->Clusters[a].weight * (temp2 - temp3);

			temp += this->Clusters[a].weight * 0.5 * log(pow(2.f * PI_GREEK * 2.71828, (float)this->Clusters[a].Mean.size()) * this->Clusters[a].Abs_Deter_Cov);
		}

		bound.second += temp;
		bound.first -= temp;
		return bound;
	}
}
