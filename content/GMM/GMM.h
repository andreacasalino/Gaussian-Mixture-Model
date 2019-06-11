//author: Andrea Casalino
//mail:andrecasa91@gmail.com

#pragma once
#ifndef _GMM__H__
#define _GMM__H__

#include <list>
#include <Eigen\Dense>

class Gaussian_Mixture_Model {
public:
	struct Train_set {
		Train_set(const Train_set&) { abort(); };
		Train_set(const std::string& file_to_read);
		Train_set(const std::list<Eigen::VectorXf>& samples) { this->__import_samples(samples); };

		struct Sample_handler;
		size_t size_X() const { return this->Samples.front().size(); };
		size_t size_set() const { return this->Samples.size(); };
		void get_samples(std::list<Eigen::VectorXf>* result) { *result = this->Samples; };
	private:
		void __import_samples(const std::list<Eigen::VectorXf>& s);

		std::list<Eigen::VectorXf> Samples;
	};

	class K_means {
	public:
		static void do_clustering(std::list<std::list<Eigen::VectorXf*>>* clusters, std::list<Eigen::VectorXf>& Samples, const size_t& N_means);
	};

	Gaussian_Mixture_Model(const std::string& train_set) { Train_set temp(train_set); this->EM_train(temp); };
	Gaussian_Mixture_Model(const std::string& train_set, const size_t& N_cluster) { Train_set temp(train_set); this->__EM_train(temp, N_cluster, NULL); };
	Gaussian_Mixture_Model(Train_set&   train_set) { this->EM_train(train_set); };
	Gaussian_Mixture_Model(Train_set&   train_set, const size_t& N_cluster) { this->__EM_train(train_set, N_cluster, NULL); };
	Gaussian_Mixture_Model(const Gaussian_Mixture_Model& to_clone);
	
	// build a random GMM with the specified number of clusters
	Gaussian_Mixture_Model(const size_t& N_clusters, const size_t& dimension_size);

// methods

	//when setting N_clusters=0, the optimal number of clusters is computed automatically
	void EM_train(std::string& train_set, const size_t& N_clusters = 0, std::list<float>* likelihood_story = NULL)
	{ 
		Train_set temp(train_set);
		this->EM_train(temp, N_clusters, likelihood_story);
	};
	//when setting N_clusters=0, the optimal number of clusters is computed automatically
	void EM_train(Train_set&   train_set, const size_t& N_clusters = 0, std::list<float>* likelihood_story = NULL);

	void Eval_log_density(float* gmm_density, Eigen::VectorXf& X);
	void Classify(Eigen::VectorXf* label_density, Eigen::VectorXf& X);
	void Classify(std::list<Eigen::VectorXf>* label_density, std::list<Eigen::VectorXf>& X);
	void Get_samples(std::list<Eigen::VectorXf>* samples, const size_t& NUmber_of_samples);

	void get_parameters(std::list<float>* weights, std::list<Eigen::VectorXf>* Means, std::list<Eigen::MatrixXf>* Covariances);
private:
	struct cluster {
		Eigen::VectorXf   Mean;
		Eigen::MatrixXf   Covariance;
		Eigen::MatrixXf   Inverse_Cov;
		float			  Abs_Deter_Cov;
		float			  weight;
	};

	void __EM_train(Train_set&   train_set, const size_t& N_clusters, std::list<float>* likelihood_story);
	void __eval_log_density(float* den, Eigen::VectorXf& X);
// data
	std::list<cluster> Clusters;
};

#endif