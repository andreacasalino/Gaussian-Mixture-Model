/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <TrainSet.h>
#include <GaussianDistribution.h>
#include <components/DivergenceAware.h>
#include <components/DrawSamplesCapable.h>
#include <components/LogDensityAware.h>

namespace gauss::gmm {
	class GaussianMixtureModel : public DivergenceAware<GaussianMixtureModel>,
								 public DrawSamplesCapable,
								 public LogDensityAware {
	public:
		struct Cluster {
			double weight;
			GaussianDistribution distribution;
		};
		// weights are interally normalized
		GaussianMixtureModel(const std::vector<Cluster>& clusters);

		/** @brief The GMM is be built using the passed train set using the expectation maximization algorithm.
		 * The initial guess used to cluster the samples, might be obtained using the K means or directly specified, according
		 * to the values put in TrainInfo.
		 * @param[in] the number of clusters to consider for training
		 * @param[in] the training set to use
		 * @param[in] the information used by the training process
		 */
		struct TrainInfo {
			// The maximum number of iterations considered by the expectation maximization algorithm.
			std::size_t maxIterations = 1000;
			// when passed empty is ingored and the K means is used for building the initial guess
			std::vector<std::size_t> initialLabeling = {};
		};
		// not really sure it should be placed here
		GaussianMixtureModel(const std::size_t& N_clusters, const TrainSet& train_set, const TrainInfo& info = TrainInfo());

		/** @brief Perform classification of a specified input.
		 * Numbers in the vector returned are the probabilities that X is coming from the corresponding cluster in the model.
		 * @param[in] the point for which the classification must be done
		 * @param[out] the classification probabilities
		 */
		Eigen::VectorXd Classify(const Eigen::VectorXd& point) const;

		std::vector<Eigen::VectorXd>
			drawSamples(const std::size_t samples) const override;

		double evaluateLogDensity(const Eigen::VectorXd& point) const override;

		double evaluateKullbackLeiblerDivergence(
			const GaussianMixtureModel& other) const override {
			throw 0;
			return 0.0;
		};

		/** @brief Estimate the Kullback-Leibler divergence of this model w.r.t. the one passed as input.
		 * Since the exact compuation is not possible, a Monte carlo approach is followed. The number of samples
		 * to assume is internally decided according to the model size.
		 * The two models must be defined in the same domain (space size).
		 * @param[in] other the model for which the divergence w.r.t. this one must be estimated
		 * @param[out] the divergence
		 */
		double getKullbackLeiblerDiergenceMonteCarlo(const GMM& other) const;

		/** @brief Estimate the Kullback-Leibler divergence of this model w.r.t. the one passed as input.
		 * Since the exact compuation is not possible, the upper and lower bound proposed in "LOWER AND UPPER BOUNDS FOR APPROXIMATION OF THE KULLBACK-LEIBLER
		 * DIVERGENCE BETWEEN GAUSSIAN MIXTURE MODELS" (Durrieu et al) is considered.
		 * The two models must be defined in the same domain (space size).
		 * @param[in] the other model for which the divergence w.r.t. this one must be estimated
		 * @param[out] <l, b>:
			l id the lower bound of the divergence
			b id the upper bound of the divergence
		 */
		std::pair<double, double> getKullbackLeiblerDiergenceEstimate(const GMM& other) const;

		const std::vector<Cluster>& getClusters() const { return clusters; };

	private:
		const std::vector<Cluster> clusters;
	};


	/** @brief Use this method to fit a GMM with an unknown number of clusters.
	 * Training is done for every possible number of clusters in N_clusters_to_try and the solution maximising the likelyhood of the training set is selected.
	 * @param[out] return the GMM model maximising the training set
	 * @param[in] the training set to assume
	 * @param[in] the number of clusters to try
	 * @param[in] the maximum number of iterations to assume for the expectation maximization (the algorithm can reach a convergence also before)
	 */
	std::unique_ptr<GaussianMixtureModel> fitOptimalModel(const TrainSet& train_set, const std::vector<std::size_t>& N_clusters_to_try, const std::size_t& Iterations = 1000);

}
