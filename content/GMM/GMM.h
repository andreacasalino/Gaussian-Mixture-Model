/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
*
* report any bug to andrecasa91@gmail.com.
 **/


#pragma once
#ifndef _GMM__H__
#define _GMM__H__

#include <vector>
#include <list>
#include <Eigen\Dense> //The Eigen library mus tbe included for using Gaussian_Mixture_Model



#define INVALID_TRAINING_SET 0
#define INVALID_INITIAL_GUESS 1
#define INVALID_NUMBER_OF_CLUSTERS 2
#define INVALID_INPUT 3
#define INVALID_GMM_PARAMETERS 4



class Gaussian_Mixture_Model {
public:

	/** \brief This class handles the training set to consider for training GMM models
	*/
	struct Train_set {
		Train_set(const Train_set& other) : Samples(other.Samples) { };
		/** \brief Import the training set from a textual file.
		* \details Every line of the file is a sample. Clearly, all the rows must have the same number of elements
		* @param[in] file_to_read The location of the file to import (it can be a relative or an absolute path)
		*/
		Train_set(const std::string& file_to_read);
		/** \brief Build a training set by cloning the samples passed as input 
		* \details Every element of the list is a sample. Clearly, all the element must have the same size
		* @param[in] Samples The list of samples characterizing the training set to build
		*/
		Train_set(const std::list<Eigen::VectorXf>& Samples);
		/** \brief Similar to Train_set::Train_set(const std::list<Eigen::VectorXf>& Samples),
		considering a vector as input.
		*/
		Train_set(const std::vector<Eigen::VectorXf>& Samples);

		/** \brief Append the samples of another set into this set.
		\details The initial guess (see Train_set::Set_initial_guess(const std::list<size_t>& cluster_initial_guess) ) is invalidated.
		* @param[in] to_append The training set whose samples must be appended to this set
		*/
		void append(const Train_set& to_append) { this->__append_samples(to_append.Samples); }; 

		/** \brief Use this to define an inital clustering to assume for the samples.
		\details You can optionally use this method to inform the Expectation maximization algorithm to start the training steps from a specified 
		set of initial clusters. cluster_initial_guess must be a list with a size equal to the number of samples in this set. 
		It specifyes the cluster to consider for every sample as initial guess. 
		For example : {0,0,1,1,2}, prescribes to consider as initial guess to put the first two samples in the first cluster, the third and the fourth into the second and the last into the third.
		* @param[in] cluster_initial_guess list of indexes to consider as initial guess
		*/
		void Set_initial_guess(const std::list<size_t>& cluster_initial_guess);

		/** \brief Get the list of samples of this set.
		*/
		const std::vector<Eigen::VectorXf>*									Get_Samples() const { return &this->Samples; };
		/** \brief Get the initial clustering to consider for the data
		\details It's used by the Expectation maximization algorithm for training a GMM.
		*/
		const std::list<std::list<const Eigen::VectorXf*>>&		Get_guess() const { return this->initial_guess_clusters; };
		/** \brief Get the sample space size, i.e. the size of each sample in this set.
		*/
		size_t																							Get_Sample_size() const { return (size_t)this->Samples.front().size(); };
		/** \brief  Get the number of samples in this set
		*/
		size_t																							Get_Samples_number() const { return this->Samples.size(); };

	private:
		template<typename Container>
		void						__append_samples(const Container& s) {

			this->Samples.reserve(s.size() + this->Samples.size());

			size_t D;
			auto it = s.begin();
			if (this->Samples.empty()) {
				D = it->size();
				this->Samples.push_back(*it);
				it++;
			}
			else D = this->Samples.front().size();

			for (it; it != s.end(); it++) {
				if (it->size() != D)
					throw INVALID_TRAINING_SET;
				this->Samples.push_back(*it);
			}

		};
		// data
		std::vector<Eigen::VectorXf>									Samples;
		std::list<std::list<const Eigen::VectorXf*>>			initial_guess_clusters;
	};

	/** \brief K means clustering .
	* \details Clusters are initialized with the Forgy method (https://en.wikipedia.org/wiki/K-means_clustering).
	Clusters are initialized with this method when training GMM models with the expectation maximization algorithm.
	* @param[in] Samples the samples to clusterize
	* @param[in] N_cluster the number of clusters to consider
	* @param[in] Iterations the maximum number of iterations to assume (the algorithm can reach a convergence also before)
	* @param[out] clusters results of the K means clustering. Every element in the list is a cluster, represented by the list of samples 
	put in that cluster (const pointers points to elements in Samples).
	*/
	static void K_means_clustering(std::list<std::list<const Eigen::VectorXf*>>* clusters, const Train_set& Samples, const size_t& N_cluster, const size_t& Iterations = 1000);

	struct cluster {
		Eigen::VectorXf		Mean;
		Eigen::MatrixXf		Covariance;
		Eigen::MatrixXf		Inverse_Cov;
		float								Abs_Deter_Cov;
		float								weight;
	};

	/** \brief A GMM is built using the expectation maximization on the samples in the passed train set.
	* @param[in] N_cluster the number of clusters to consider for training
	* @param[in] train_set the training set to assume
	* @param[in] Iterations the maximum number of iterations to assume for the expecation maximization (the algorithm can reach a convergence also before)
	* @param[out] train_set_lklhood the likelihood of the training set of the obtained model
	*/
	Gaussian_Mixture_Model(const size_t& N_clusters, const Train_set& train_set,  const size_t& Iterations = 1000, float* train_set_lklhood = NULL);
	/** \brief Build a random GMM with the specified number of clusters and space size.
	* @param[in] N_cluster the number of clusters to put in the model
	* @param[in] dimension_size the size of the space of the GMM to create
	*/
	Gaussian_Mixture_Model(const size_t& N_clusters, const size_t& dimension_size);
	/** \brief Build a GMM model, assuming the paramaters contained in the passed matrix.
	\details Gaussian_Mixture_Model::get_parameters_as_matrix can be used to extract the parameters (in a matricial form) of a GMM model.
	The consistency of the parameters (Positiveness of covariance matrices, etc..) is interanally checked.
	* @param[in] params The parameters to assume for the model
	*/
	Gaussian_Mixture_Model(const Eigen::MatrixXf& params);
	/** \brief Use this method to fit a GMM with an unknown number of clusters.
	\details 	Training is done for every possible number of clusters in N_clusters_to_try and the solution maximising the likelyhood of the training set is selected. 
	* @param[out] return the GMM model maximising the training set
	* @param[in] train_set the training set to assume
	* @param[in] Iterations the maximum number of iterations to assume for the expecation maximization (the algorithm can reach a convergence also before)
	* @param[in] N_clusters_to_try the number of clusters to try
	* @param[out] train_set_lklhood the likelihood of the training set of best model (the one returned)
	*/
	static Gaussian_Mixture_Model Fit_optimal_model(const Train_set& train_set, const std::list<size_t>& N_clusters_to_try, const size_t& Iterations = 1000, float* train_set_lklhood = NULL); //returns likelihood of training set after fitting the model (the best one)

	/** \brief Evaluates the logarithmic density of the mixture into a specified point.
	* @param[in] X the point for which the density must be evaluated
	* @param[out] return the logarithmic value of the density
	*/
	float Get_log_density(const Eigen::VectorXf& X) const  { return this->__eval_log_density(X); };

	/** \brief Perform classification of a specified input.
	\details Numbers in label_density are the probabilities that X is coming from a certain cluster in the model.
	* @param[in] X the point for which the classification must be done
	* @param[out] label_density the classification probabilities 
	*/
	void Classify(Eigen::VectorXf* label_density, Eigen::VectorXf& X) const;
	/** \brief Similar to Gaussian_Mixture_Model::Classify(Eigen::VectorXf* label_density, Eigen::VectorXf& X)
	\details Here the classification is done for a certain list of inputs.
	* @param[in] X the points for which the classification must be done
	* @param[out] label_density the classification probabilities (one vector for each element in X)
	*/
	void Classify(std::list<Eigen::VectorXf>* label_density, std::list<Eigen::VectorXf>& X) const;

	/** \brief Draw samples from the distribution represented by this GMM.
	* @param[in] Number_of_samples the number of samples to draw
	* @param[out] samples
	*/
	void Get_samples(std::list<Eigen::VectorXf>* samples, const size_t& Number_of_samples) const;

	/** \brief Estimate the Kullback-Leibler divergence of this model w.r.t. the one passed as input
	\details Since the exact compuation is not possible, a Monte carlo approach is followed. The number of samples
	to assume is internally decided according to the model size.
	The two models must be defined in the same domain (space size).
	* @param[in] other the model for which the divergence w.r.t. this one must be estimated
	* @param[out] return the divergence
	*/
	float Get_KULLBACK_LEIBLER_divergence_MonteCarlo(const Gaussian_Mixture_Model& other) const;
	/** \brief Estimate the Kullback-Leibler divergence of this model w.r.t. the one passed as input
	\details Since the exact compuation is not possible, the upper and lower bound proposed in "LOWER AND UPPER BOUNDS FOR APPROXIMATION OF THE KULLBACK-LEIBLER
DIVERGENCE BETWEEN GAUSSIAN MIXTURE MODELS" (Durrieu et al) is considered. 
	The two models must be defined in the same domain (space size).
	* @param[in] other the model for which the divergence w.r.t. this one must be estimated
	* @param[out] upper_bound upper bound of the divergence
	* @param[out] lower_bound lower bound of the divergence
	*/
	void Get_KULLBACK_LEIBLER_divergence_estimate(const Gaussian_Mixture_Model& other, float* upper_bound, float* lower_bound) const;

	/** \brief Get a copy of the model parameters
	*/
	const std::vector<cluster>&					get_parameters()  const  { return this->Clusters; };
	/** \brief Similar to get_parameters.
	\details Here the parameters are put in Matrix of numbers.
	*/
	Eigen::MatrixXf										get_parameters_as_matrix()  const;
	/** \brief Similar to get_parameters_as_matrix.
	\details Here the parameters are put in a JSON.
	*/
	std::string															get_paramaters_as_JSON() const;
	/** \brief Get the number of clusters in the model
	*/
	size_t															get_clusters_number()  const { return this->Clusters.size(); };
	/** \brief Get the size of the domain in which the mixture is defined
	*/
	size_t															get_Space_size()  const { return this->Clusters.front().Mean.size(); };


private:

	float __EM_train(const Train_set&   train_set, const size_t& N_clusters, const size_t& Iterations);
	void __append_cluster(const float& w, const Eigen::VectorXf& M, const Eigen::MatrixXf& C);
	float __eval_log_density(const Eigen::VectorXf& X) const;

// data
	std::vector<cluster>		Clusters;
};

#endif