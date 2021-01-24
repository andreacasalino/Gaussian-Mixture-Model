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
		/** \brief A GMM is built using the expectation maximization on the samples in the passed train set.
		* @param[in] N_cluster the number of clusters to consider for training
		* @param[in] train_set the training set to assume
		* @param[in] Iterations the maximum number of iterations to assume for the expecation maximization (the algorithm can reach a convergence also before)
		* @param[out] train_set_lklhood the likelihood of the training set of the obtained model
		*/
		struct TrainInfo {
			std::size_t maxIterations = 1000;
			std::list<std::size_t> initialLabeling; // when passed empty is ingored
		};
		GMM(const size_t& N_clusters, const TrainSet& train_set, const TrainInfo& info = TrainInfo());

		/** \brief Build a random GMM with the specified number of clusters and space size.
		* @param[in] N_cluster the number of clusters to put in the model
		* @param[in] dimension_size the size of the space of the GMM to create
		*/
		GMM(const size_t& N_clusters, const size_t& dimension_size);

		/** \brief Build a GMM model, assuming the paramaters contained in the passed matrix.
		\details Gaussian_Mixture_Model::get_parameters_as_matrix can be used to extract the parameters (in a matricial form) of a GMM model.
		The consistency of the parameters (Positiveness of covariance matrices, etc..) is interanally checked.
		* @param[in] params The parameters to assume for the model
		*/
		GMM(const M& params);

		/** \brief Use this method to fit a GMM with an unknown number of clusters.
		\details 	Training is done for every possible number of clusters in N_clusters_to_try and the solution maximising the likelyhood of the training set is selected.
		* @param[out] return the GMM model maximising the training set
		* @param[in] train_set the training set to assume
		* @param[in] Iterations the maximum number of iterations to assume for the expecation maximization (the algorithm can reach a convergence also before)
		* @param[in] N_clusters_to_try the number of clusters to try
		* @param[out] train_set_lklhood the likelihood of the training set of best model (the one returned)
		*/
		static GMM fitOptimalModel(const TrainSet& train_set, const std::vector<size_t>& N_clusters_to_try, const size_t& Iterations = 1000);

		/** \brief Evaluates the logarithmic density of the mixture into a specified point.
		* @param[in] X the point for which the density must be evaluated
		* @param[out] return the logarithmic value of the density
		*/
		double getLogDensity(const V& X) const;

		double getLogLikelihood(const TrainSet& train_set);

		/** \brief Perform classification of a specified input.
		\details Numbers in label_density are the probabilities that X is coming from a certain cluster in the model.
		* @param[in] X the point for which the classification must be done
		* @param[out] label_density the classification probabilities
		*/
		V Classify(const V& X) const;

		/** \brief Draw samples from the distribution represented by this GMM.
		* @param[in] Number_of_samples the number of samples to draw
		* @param[out] samples
		*/
		std::list<V> drawSamples(const size_t& Number_of_samples) const;

		/** \brief Estimate the Kullback-Leibler divergence of this model w.r.t. the one passed as input
		\details Since the exact compuation is not possible, a Monte carlo approach is followed. The number of samples
		to assume is internally decided according to the model size.
		The two models must be defined in the same domain (space size).
		* @param[in] other the model for which the divergence w.r.t. this one must be estimated
		* @param[out] return the divergence
		*/
		double getKullbackLeiblerDiergenceMonteCarlo(const GMM& other) const;

		/** \brief Estimate the Kullback-Leibler divergence of this model w.r.t. the one passed as input
		\details Since the exact compuation is not possible, the upper and lower bound proposed in "LOWER AND UPPER BOUNDS FOR APPROXIMATION OF THE KULLBACK-LEIBLER
	DIVERGENCE BETWEEN GAUSSIAN MIXTURE MODELS" (Durrieu et al) is considered.
		The two models must be defined in the same domain (space size).
		* @param[in] other the model for which the divergence w.r.t. this one must be estimated
		* @param[out] upper_bound upper bound of the divergence
		* @param[out] lower_bound lower bound of the divergence
		*/
		std::pair<double, double> getKullbackLeiblerDiergenceEstimate(const GMM& other) const;

		/** \brief Get a copy of the model parameters
		*/
		inline const std::vector<GMMcluster>& getParameters()  const { return this->Clusters; };

		/** \brief Similar to get_parameters.
		\details Here the parameters are put in Matrix of numbers.
		*/
		M getMatrixParameters()  const;

		inline const std::vector<GMMcluster> getClusters() const { return this->Clusters; };

		//scale();

	private:
		double ExpectationMaximization(const TrainSet& train_set, const size_t& N_clusters, const TrainInfo& info);
		void  appendCluster(const double& w, const V& mean, const M& cov);

	// data
		std::vector<GMMcluster> Clusters;
	};
}

#endif
