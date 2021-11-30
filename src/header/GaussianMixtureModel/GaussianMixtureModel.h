/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianUtils/GaussianDistribution.h>

namespace gauss::gmm {
	struct Cluster {
		double weight;
		std::unique_ptr<GaussianDistribution> distribution;
	};

	class GaussianMixtureModel : public DivergenceAware<GaussianMixtureModel>,
								 public DrawSamplesCapable,
								 public LogDensityAware,
								 public StateSpaceSizeAware {
	public:
		// weights are interally normalized
		GaussianMixtureModel(const std::vector<Cluster>& clusters);

		GaussianMixtureModel(const double weight, const gauss::GaussianDistribution& distribution);

		GaussianMixtureModel(const GaussianMixtureModel& o);
		GaussianMixtureModel& operator=(const GaussianMixtureModel& o);

		GaussianMixtureModel(GaussianMixtureModel&& o) = default;
		GaussianMixtureModel& operator=(GaussianMixtureModel&& o) = default;

		std::size_t getStateSpaceSize() const override { return clusters.front().distribution->getStateSpaceSize(); }

		/** 
		 * @param[in] the weight of the cluster to add (is internally re-scaled)
		 * @param[in] the distribution of the cluster to add
		 * @throw in case the weight is negative
		 */
		void addCluster(const double weight, const gauss::GaussianDistribution& distribution);

		/** @brief Use this method to fit a GMM with an unknown number of clusters.
		 * Training is done for every possible number of clusters in N_clusters_to_try and the solution maximising the likelyhood of the training set is selected.
		 * @param[out] return the GMM model maximising the training set
		 * @param[in] the training set to assume
		 * @param[in] the number of clusters to try
		 * @param[in] the maximum number of iterations to assume for the expectation maximization (the algorithm can reach a convergence also before)
		 */
		static std::unique_ptr<GaussianMixtureModel> fitOptimalModel(const TrainSet& train_set, const std::vector<std::size_t>& N_clusters_to_try, const std::size_t& Iterations = 1000);

		/** @brief Perform classification of a specified input.
		 * Numbers in the vector returned are the probabilities that X is coming from the corresponding cluster in the model.
		 * @param[in] the point for which the classification must be done
		 * @param[out] the classification probabilities
		 */
		Eigen::VectorXd Classify(const Eigen::VectorXd& point) const;

		std::vector<Eigen::VectorXd>
			drawSamples(const std::size_t samples) const override;

		double evaluateLogDensity(const Eigen::VectorXd& point) const override;

		void setMonteCarloTrials(const std::size_t trials) { monte_carlo_trials = trials; };

		/** @brief Estimate the Kullback-Leibler divergence of this model w.r.t. the one passed as input.
		 * Since the exact compuation is not possible, a Monte carlo approach is followed. The number of samples
		 * to assume is internally decided according to the model size.
		 * The two models must be defined in the same domain (space size).
		 * @param[in] other the model for which the divergence w.r.t. this one must be estimated
		 * @param[out] the divergence
		 */
		double evaluateKullbackLeiblerDivergence(
			const GaussianMixtureModel& other) const override;

		/** @brief Estimate the Kullback-Leibler divergence of this model w.r.t. the one passed as input.
		 * Since the exact compuation is not possible, the upper and lower bound proposed in "LOWER AND UPPER BOUNDS FOR APPROXIMATION OF THE KULLBACK-LEIBLER
		 * DIVERGENCE BETWEEN GAUSSIAN MIXTURE MODELS" (Durrieu et al) is considered.
		 * The two models must be defined in the same domain (space size).
		 * @param[in] the other model for which the divergence w.r.t. this one must be estimated
		 * @param[out] <l, b>:
			l id the lower bound of the divergence
			b id the upper bound of the divergence
		 */
		std::pair<double, double> estimateKullbackLeiblerDivergence(const GaussianMixtureModel& other) const;

		const std::vector<Cluster>& getClusters() const { return clusters; };

	private:
		std::size_t monte_carlo_trials = 500;
		std::vector<double> original_clusters_weights; // before re-scaling
		std::vector<Cluster> clusters;
	};
}
