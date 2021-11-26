/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <header/GMM.h>
#include <random>
using namespace std;

namespace gmm {
	class GaussianSampler {
	public:
		GaussianSampler(const V& Mean, const M& Sigma) : gauss_iso(0.0, 1.0) {
			this->Trasl = Mean;
			Eigen::LLT<M> lltOfCov(Sigma);
			this->Rot = lltOfCov.matrixL();
		}

		V getSample() {
			V sample(this->Trasl.size());
			for (std::size_t k = 0; k < (std::size_t)this->Trasl.size(); ++k)
				sample(k) = gauss_iso(generator);
			sample = this->Rot * sample;
			sample += this->Trasl;
			return sample;
		}

	private:
		V									Trasl;
		M									Rot;
		std::normal_distribution<double>    gauss_iso;
		default_random_engine				generator;
	};

	class DiscreteSampler {
	public:
		DiscreteSampler(const std::vector<double>& d) : distr(d) {};

		std::size_t getSample() {
			double r = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
			this->c = 0.0;
			std::size_t k = 0;
			for (auto it = this->distr.begin(); it != this->distr.end(); ++it) {
				c += *it;
				if (r <= c)
					return k;
				++k;
			}
			return this->distr.size();
		}

	private:
		std::vector<double> distr;
		double c;
	};

	std::list<V> GMM::drawSamples(const std::size_t& Number_of_samples) const {
		std::vector<double>			wDistr;
		std::vector<GaussianSampler> gSamplers;

		wDistr.reserve(this->Clusters.size());
		gSamplers.reserve(this->Clusters.size());
		for (auto it = this->Clusters.begin(); it != this->Clusters.end(); ++it) {
			wDistr.emplace_back(it->weight);
			gSamplers.emplace_back(it->Mean, it->Covariance);
		}
		DiscreteSampler dSampler(wDistr);

		std::list<V> samples;
		for (std::size_t k = 0; k < Number_of_samples; ++k) {
			samples.push_back(gSamplers[dSampler.getSample()].getSample());
		}
		return samples;
	}
}
