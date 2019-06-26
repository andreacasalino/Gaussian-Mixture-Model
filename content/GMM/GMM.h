//author: Andrea Casalino
//mail:andrecasa91@gmail.com

#pragma once
#ifndef _GMM__H__
#define _GMM__H__

#include <list>
#include <Eigen\Dense>
#include <random>

class Gaussian_sampler {
public:
	Gaussian_sampler(Eigen::VectorXf& Mean, Eigen::MatrixXf& Sigma);

	void get_sample(Eigen::VectorXf* sample);
private:
	Eigen::VectorXf						Trasl;
	Eigen::MatrixXf						Rot;
	std::normal_distribution<float>		    gauss_iso;
};

class Gaussian_Mixture_Model {
public:

	struct Train_set {
		Train_set(const Train_set& to_clone);
		Train_set(const std::string& file_to_read);
		//cluster_initial_guess is optionally used to inform the GMM trainer to start learning from a specified 
		//set of initial clusters. cluster_initial_guess is a list with the same length of samples, 
	    //specifying for every sample the cluster index. For example : {0,0,1,1,2}.
		//In the previous example it is prescribed that the first two samples are in the first cluster and the last three in the third (this
		// will be valid only for the initial guess).
		Train_set(const std::list<Eigen::VectorXf>& samples, const std::list<size_t>& cluster_initial_guess = std::list<size_t>());
		~Train_set() { if (!this->was_cloned) delete this->Samples; };

		struct Sample_handler;
		size_t size_X() const { return this->Samples->front().size(); };
		size_t size_set() const { return this->Samples->size(); };
		void get_samples(std::list<Eigen::VectorXf>* result) const { *result = *this->Samples; };
	private:
		Train_set() : was_cloned(false), Samples(new std::list<Eigen::VectorXf>()) {};

		void						__import_samples(const std::list<Eigen::VectorXf>& s);

		bool								   was_cloned;
		std::list<Eigen::VectorXf>*			   Samples;
		std::list<std::list<Eigen::VectorXf*>> initial_guess_clusters;
	};

	class K_means {
	public:
		static void do_clustering(std::list<std::list<Eigen::VectorXf*>>* clusters, std::list<Eigen::VectorXf>& Samples, const size_t& N_means);
	};

// When specifying do_trials = true, training is done for every possible number of clusters in the list = 1,2,3,....,N_clusters
// and the solution maximising the likelyhood of the training set is selected.
// Otherwise is done considering only the passed number of clusters.
// Optionally, the likelihhod story of every training is returned. When a single do_trials is set false and likelihood_story
// is passed non NULL a single list is returned describing the evolution of the likelihood for the only traininig session done
	Gaussian_Mixture_Model(const Train_set& train_set, const size_t& N_clusters, const bool& do_trials = false, std::list<std::list<float>>* likelihood_story = NULL);
// Training is done for every possible number of clusters in N_clusters_to_try 
// and the solution maximising the likelyhood of the training set is selected. 
// Optionally, the likelihhod story of every training is returned.
	Gaussian_Mixture_Model(const Train_set& train_set, const std::list<size_t>& N_clusters_to_try, std::list<std::list<float>>* likelihood_story = NULL);

	Gaussian_Mixture_Model(const Gaussian_Mixture_Model& to_clone) { this->__copy(to_clone); };
	
// build a random GMM with the specified number of clusters
	Gaussian_Mixture_Model(const size_t& N_clusters, const size_t& dimension_size);

// build a GMM passing the info taken from another GMM with GMM::get_parameters(std::list<float>* weights, std::list<Eigen::VectorXf>* Means, std::list<Eigen::MatrixXf>* Covariances)
	Gaussian_Mixture_Model(const std::list<float>& weights,const  std::list<Eigen::VectorXf>& Means,const std::list<Eigen::MatrixXf>& Covariances);
// build a GMM passing the info taken from another GMM with GMM::get_parameters(Eigen::MatrixXf* packed_params)
	Gaussian_Mixture_Model(Eigen::MatrixXf& packed_params);

// methods

// The same number of clusters determined when building this class is assumed for this new training session
	void EM_train(const Train_set&   train_set, std::list<float>* likelihood_story = NULL);

	void Eval_log_density(float* gmm_density, Eigen::VectorXf& X);
	void Classify(Eigen::VectorXf* label_density, Eigen::VectorXf& X);
	void Classify(std::list<Eigen::VectorXf>* label_density, std::list<Eigen::VectorXf>& X);
	void Get_samples(std::list<Eigen::VectorXf>* samples, const size_t& NUmber_of_samples);

	void get_parameters(std::list<float>* weights, std::list<Eigen::VectorXf>* Means, std::list<Eigen::MatrixXf>* Covariances);
	void get_parameters(Eigen::MatrixXf* packed_params);

	size_t get_clusters_number() { return this->Clusters.size(); };
	size_t get_Feature_size() { return this->Clusters.front().Mean.size(); };
private:
	struct cluster {
		cluster(const float& w, const Eigen::VectorXf& M, const Eigen::MatrixXf& C);

		Eigen::VectorXf   Mean;
		Eigen::MatrixXf   Covariance;
		Eigen::MatrixXf   Inverse_Cov;
		float			  Abs_Deter_Cov;
		float			  weight;
	};

	void __EM_train(const Train_set&   train_set, const size_t& N_clusters, std::list<float>* likelihood_story);
	void __eval_log_density(float* den, Eigen::VectorXf& X);
	void __copy(const Gaussian_Mixture_Model& to_clone);
	void __check_eig_Cov();
// data
	std::list<cluster> Clusters;
};

#endif