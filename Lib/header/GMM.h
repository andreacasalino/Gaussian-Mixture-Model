/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#ifndef GMM_GMM_H
#define GMM_GMM_H

#include "TrainSet.h"

namespace gmm {
	typedef Eigen::MatrixXd M;

	struct GMMcluster {
		V		Mean;
		M		Covariance;
		M		Inverse_Cov;
		double	Abs_Deter_Cov;
		double	weight;
	};

	class GMM {
	public:
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
			std::list<std::size_t> initialLabeling;
		};
		GMM(const std::size_t& N_clusters, const TrainSet& train_set, const TrainInfo& info = TrainInfo());

		/** @brief Build a random GMM with the specified number of clusters and space size.
		 * @param[in] the number of clusters to put in the model
		 * @param[in] the size of the space of the GMM to create
		 */
		GMM(const std::size_t& N_clusters, const std::size_t& dimension_size);

		/** @brief Build a GMM model, assuming the paramaters contained in the passed matrix.
		 * GMM::getMatrixParameters() can be used to extract the parameters (in a matricial form) of a GMM model.
		 * The consistency of the parameters (Positiveness of covariance matrices, etc..) is interanally checked.
		 * @param[in] The parameters to assume for the model
		 */
		GMM(const M& params);

		/** @brief Use this method to fit a GMM with an unknown number of clusters.
		 * Training is done for every possible number of clusters in N_clusters_to_try and the solution maximising the likelyhood of the training set is selected.
		 * @param[out] return the GMM model maximising the training set
		 * @param[in] the training set to assume
		 * @param[in] the number of clusters to try
		 * @param[in] the maximum number of iterations to assume for the expectation maximization (the algorithm can reach a convergence also before)
		 */
		static GMM fitOptimalModel(const TrainSet& train_set, const std::vector<std::size_t>& N_clusters_to_try, const std::size_t& Iterations = 1000);

		/** @brief Evaluates the logarithmic density of the mixture into a specified point.
		 * @param[in] the point for which the density must be evaluated
		 * @param[out] the logarithmic value of the density
		 */
		double getLogDensity(const V& X) const;

		/** @brief Evaluates the logarithmic likelihood of a specified set of samples
		 * @param[in] the samples for which the likelihood must be evaluated
		 * @param[out] the logarithmic value of the likelihood
		 */
		double getLogLikelihood(const TrainSet& train_set);

		/** @brief Perform classification of a specified input.
		 * Numbers in the vector returned are the probabilities that X is coming from the corresponding cluster in the model.
		 * @param[in] the point for which the classification must be done
		 * @param[out] the classification probabilities
		 */
		V Classify(const V& X) const;

		/** @brief Draw samples from the distribution represented by this GMM.
		 * @param[in] the number of samples to draw
		 * @param[out] the drawn samples
		 */
		std::list<V> drawSamples(const std::size_t& Number_of_samples) const;

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

		inline const std::vector<GMMcluster>& getParameters()  const { return this->Clusters; };

		/** @brief Similar to getParameters.
		 * Here the parameters are put in Matrix of numbers.
		 */
		M getMatrixParameters()  const;

		inline const std::vector<GMMcluster> getClusters() const { return this->Clusters; };

		//scale();

	private:
		double ExpectationMaximization(const TrainSet& train_set, const std::size_t& N_clusters, const TrainInfo& info);
		void  appendCluster(const double& w, const V& mean, const M& cov);

	// data
		std::vector<GMMcluster> Clusters;
	};
}

#endif
