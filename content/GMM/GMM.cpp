//author: Andrea Casalino
//mail:andrecasa91@gmail.com

#include "GMM.h"
#include <fstream>
#include <string>
#include <sstream>
using namespace std;
#include <Eigen\Cholesky>
using namespace Eigen;

#define log_2_pi logf(2.f * 3.141592f)
default_random_engine generator;
#define MAX_ITER_EM 1000

void parse_slices(list<string>* slices, string& line) {

	std::istringstream iss(line);
	slices->clear();
	while (true) {
		if (iss.eof()) {
			break;
		}
		slices->push_back(std::string());
		iss >> slices->back();
	}

}

void slices_2_Vector(VectorXf* V, const list<string>& slices) {

	if (slices.empty()) abort();

	*V = VectorXf(slices.size());
	auto it = slices.begin();
	for (size_t k = 0; k < slices.size(); k++) {
		(*V)(k) = (float)atof(it->c_str());
		it++;
	}

}

void Gaussian_Mixture_Model::Train_set::__import_samples(const std::list<Eigen::VectorXf>& s) {

	if (s.empty()) abort();
	auto it = s.begin();
	size_t D = it->size();
	this->Samples->push_back(*it);
	it++;
	for (it; it != s.end(); it++) {
		if (it->size() != D) 
			abort();		
		this->Samples->push_back(*it);
	}

};

Gaussian_Mixture_Model::Train_set::Train_set(const Train_set& to_clone) {

	this->was_cloned = true;
	this->Samples = to_clone.Samples;

}

Gaussian_Mixture_Model::Train_set::Train_set(const std::string& file_to_read) : Train_set() {

	ifstream f(file_to_read);

	if (!f.is_open()) abort();

	string buffer;
	list<string> slices;
	list<string>::iterator it_s;
	list<VectorXf> raw;
	while (!f.eof()) {
		getline(f, buffer);
		parse_slices(&slices, buffer);

		raw.push_back(VectorXf());
		slices_2_Vector(&raw.back(), slices);
	}
	f.close();

	this->__import_samples(raw);

}

Gaussian_Mixture_Model::Train_set::Train_set(const std::list<Eigen::VectorXf>& samples) : Train_set() {

	this->__import_samples(samples); 

};

struct Gaussian_Mixture_Model::Train_set::Sample_handler {
	static list<VectorXf>* get_samples(Gaussian_Mixture_Model::Train_set& set) { return set.Samples; };
};







void get_mean(VectorXf* Mean, list<VectorXf*>* l) {

	if (l->empty()) abort();

	auto it = l->begin();
	*Mean = *l->front();
	it++;
	for (it; it != l->end(); it++)
		*Mean += *(*it);
	*Mean = (1.f / (float)l->size()) * (*Mean);

}

void recompute_means(list<VectorXf>* Means, list<list<VectorXf*>>& clusters) {

	Means->clear();
	for (auto it = clusters.begin(); it != clusters.end(); it++) {
		Means->push_back(VectorXf());
		get_mean(&Means->back(), &(*it));
	}

}

bool check_are_the_same(const list<list<VectorXf*>>& a, const list<list<VectorXf*>>& b) {

	auto it_b = b.begin();
	auto it_a = a.begin();
	auto it_b_b = it_b->begin();
	auto it_a_a = it_a->begin();
	for (it_a; it_a != a.end(); it_a++) {
		if (it_a->size() != it_b->size()) return false;

		it_b_b = it_b->begin();
		for (it_a_a = it_a->begin(); it_a_a != it_a->end(); it_a_a++) {
			if (*it_a_a != *it_b_b)
				return false;
			it_b_b++;
		}

		it_b++;
	}
	return true;

}

void Gaussian_Mixture_Model::K_means::do_clustering(std::list<std::list<Eigen::VectorXf*>>* clusters, std::list<Eigen::VectorXf>& Samples, const size_t& N_means) {

	if (N_means == 0) abort();
	if (N_means > Samples.size()) abort();

	clusters->clear();
	auto it_s = Samples.begin();
	list<VectorXf> Means;
	Means.push_back(*it_s);
	it_s++;
	clusters->push_back(list<VectorXf*>());
	for (size_t km = 1; km < N_means; km++) {
		clusters->push_back(list<VectorXf*>());
		Means.push_back(*it_s);
		it_s++;
	}
	auto it = Means.begin();

	auto it_c = clusters->begin();
	float dist_min, temp;
	size_t pos_nearest, kk;
	list<list<VectorXf*>> old_clustering;
	for (size_t k = 0; k < 1000; k++) {
		for (it_c = clusters->begin(); it_c != clusters->end(); it_c++)
			it_c->clear();

		for (it_s = Samples.begin(); it_s != Samples.end(); it_s++) {
			it = Means.begin();
			pos_nearest = 0;
			dist_min = (*it_s - *it).squaredNorm(), temp;
			it++;
			kk = 1;
			for (it; it != Means.end(); it++) {
				temp = (*it_s - *it).squaredNorm();
				if (temp < dist_min) {
					dist_min = temp;
					pos_nearest = kk;
				}
				kk++;
			}

			it_c = clusters->begin();
			advance(it_c, pos_nearest);
			it_c->push_back(&(*it_s));
		}

		recompute_means(&Means, *clusters);

		if (!old_clustering.empty()) {
			if (check_are_the_same(old_clustering, *clusters))
				return;
		}

		old_clustering = *clusters;
		//system("ECHO iter");
	}

}







void normalize_list(list<float>& l) {

	float s = 0.f;
	auto it = l.begin();
	for (it; it != l.end(); it++)
		s += *it;

	if (abs(s) < 1e-7) return;

	for (it = l.begin(); it != l.end(); it++)
		*it = *it / s;

}

void eval_Normal_Log_density(float* den, VectorXf& Mean, MatrixXf& invCov, float& absDetCov, VectorXf& X) {

	*den = (X - Mean).transpose() * invCov * (X - Mean);
	*den += X.size() * log_2_pi;
	*den += logf(absDetCov);

	*den *= -0.5f;

}

void invert_symm_positive(MatrixXf* Sigma_inverse, MatrixXf& Sigma) {

	//LLT<MatrixXf> lltOfCov(Sigma);
	//MatrixXf L(lltOfCov.matrixL());
	//*Sigma_inverse = L * L.transpose();

	*Sigma_inverse = Sigma.inverse();

}

Gaussian_Mixture_Model::cluster::cluster(const float& w, const Eigen::VectorXf& M, const Eigen::MatrixXf& C):
	weight(w), Mean(M), Covariance(C) {

	invert_symm_positive(&this->Inverse_Cov , Covariance);
	this->Abs_Deter_Cov = abs(Covariance.determinant());

}

void Gaussian_Mixture_Model::__eval_log_density(float* den, VectorXf& X) {

	*den = 0.f;
	float temp;
	for (auto it_cl = this->Clusters.begin(); it_cl != this->Clusters.end(); it_cl++) {
		eval_Normal_Log_density(&temp, it_cl->Mean, it_cl->Inverse_Cov, it_cl->Abs_Deter_Cov, X);
		*den += expf(logf(it_cl->weight) + temp);
	}
	*den = logf(*den);

}

void Gaussian_Mixture_Model::Eval_log_density(float* gmm_density, VectorXf& X) {

	this->__eval_log_density(gmm_density, X);

}

void Gaussian_Mixture_Model::Classify(VectorXf* label_density, VectorXf& X) {

	*label_density = VectorXf(this->Clusters.size());
	size_t k = 0;
	float temp;
	for (auto it = this->Clusters.begin(); it != this->Clusters.end(); it++) {
		eval_Normal_Log_density(&temp, it->Mean, it->Inverse_Cov, it->Abs_Deter_Cov, X);
		(*label_density)(k) = expf(temp);
		k++;
	}

	*label_density *= (1.f / label_density->sum());

}

Gaussian_sampler::Gaussian_sampler(VectorXf& Mean, MatrixXf& Sigma) : gauss_iso(0.f, 1.f) {

	this->Trasl = Mean;
	LLT<MatrixXf> lltOfCov(Sigma);
	this->Rot = lltOfCov.matrixL();

}

void Gaussian_sampler::get_sample(VectorXf* sample) {

	*sample = VectorXf(this->Trasl.size());
	for (size_t k = 0; k < (size_t)this->Trasl.size(); k++)
		(*sample)(k) = gauss_iso(generator);

	*sample = this->Rot * *sample;
	*sample += this->Trasl;

}

size_t sample_from_discrete(const list<float>& w) {

	float r = (float)rand() / (float)RAND_MAX;
	float c = 0.f;
	size_t k = 0;
	for (auto it = w.begin(); it != w.end(); it++) {
		c += *it;
		if (r <= c) 
			return k;
		k++;
	}
	return w.size();

}

void Gaussian_Mixture_Model::Get_samples(std::list<Eigen::VectorXf>* samples, const size_t& NUmber_of_samples) {

	list<float>					w_as_list;
	list<Gaussian_sampler*>		Samplers;

	for (auto it = this->Clusters.begin(); it != this->Clusters.end(); it++) {
		w_as_list.push_back(it->weight);
		Samplers.push_back(new Gaussian_sampler(it->Mean, it->Covariance));
	}

	samples->clear();
	VectorXf X;
	size_t sampled_pos;
	list<Gaussian_sampler*>::iterator it_S;
	for (size_t k = 0; k < NUmber_of_samples; k++) {
		sampled_pos = sample_from_discrete(w_as_list);
		it_S = Samplers.begin();
		advance(it_S, sampled_pos);
		(*it_S)->get_sample(&X);
		samples->push_back(VectorXf());
		samples->back() = X;
	}

	for (auto it = Samplers.begin(); it != Samplers.end(); it++)
		delete *it;

}

void Gaussian_Mixture_Model::EM_train(const Train_set&   train_set, std::list<float>* likelihood_story) {

	this->__EM_train(train_set, this->Clusters.size(), likelihood_story);

}

void Gaussian_Mixture_Model::__EM_train(const Train_set&   train_set, const size_t& N_clusters, list<float>* likelihood_story) {

	if (N_clusters == 0)
		abort();

	Train_set t_temp(train_set);

	if (likelihood_story != NULL)
		likelihood_story->clear();

	auto Samples = Gaussian_Mixture_Model::Train_set::Sample_handler::get_samples(t_temp);

	auto it_s = Samples->begin();
	if (this->Clusters.size() != N_clusters) {
	//initialize clusters with k-means
		list<list<VectorXf*>> clst;

		K_means::do_clustering(&clst, *Samples, N_clusters);
		this->Clusters.clear();
		VectorXf M_temp(clst.front().front()->size());
		MatrixXf C_temp(clst.front().front()->size(), clst.front().front()->size());
		for (auto it = clst.begin(); it != clst.end(); it++) {
			get_mean(&M_temp, &(*it));
			C_temp.setZero();
			for (it_s = Samples->begin(); it_s != Samples->end(); it_s++) {
				C_temp += (*it_s - M_temp) * (*it_s - M_temp).transpose();
			}
			C_temp *= 1.f / (float)Samples->size();

			this->Clusters.push_back(cluster(1.f / (float)clst.size(), M_temp, C_temp));
		}
	}

//EM loop
	int R = (int)Samples->size();
	int C = (int)this->Clusters.size();
	auto it_c = this->Clusters.begin();
	MatrixXf gamma(R, C);
	int r, c;
	float temp;
	VectorXf n(C);
	float old_lkl = FLT_MAX, new_lkl;
	for (int k = 0; k < MAX_ITER_EM; k++) {
		it_c = this->Clusters.begin();
		for (c = 0; c < C; c++) {
			it_s = Samples->begin();
			for (r = 0; r < R; r++) {
				eval_Normal_Log_density(&temp, it_c->Mean, it_c->Inverse_Cov, it_c->Abs_Deter_Cov, *it_s);
				gamma(r, c) = it_c->weight * expf(temp);
				it_s++;
			}
			it_c++;
		}

		for (r = 0; r < R; r++)
			gamma.row(r) = (1.f / gamma.row(r).sum()) * gamma.row(r);

		for (c = 0; c < C; c++)
			n(c) = gamma.col(c).sum();

		c = 0;
		for (it_c = this->Clusters.begin(); it_c != this->Clusters.end(); it_c++) {
			it_c->weight = n(c) / (float)R;

			it_c->Mean.setZero();
			r = 0;
			for (it_s = Samples->begin(); it_s != Samples->end(); it_s++) {
				it_c->Mean += gamma(r, c) * *it_s;
				r++;
			}
			it_c->Mean *= 1.f / n(c);

			it_c->Covariance.setZero();
			r = 0;
			for (it_s = Samples->begin(); it_s != Samples->end(); it_s++) {
				it_c->Covariance += gamma(r,c) * (*it_s - it_c->Mean) * (*it_s - it_c->Mean).transpose();
				r++;
			}
			it_c->Covariance *= 1.f / n(c);
			c++;

			invert_symm_positive(&it_c->Inverse_Cov , it_c->Covariance);
			it_c->Abs_Deter_Cov = abs(it_c->Covariance.determinant());
		}

		new_lkl = 0.f;
		for (it_s = Samples->begin(); it_s != Samples->end(); it_s++) {
			this->__eval_log_density(&temp, *it_s);
			new_lkl += temp;
		}

		if (likelihood_story != NULL)
			likelihood_story->push_back(new_lkl);

		if (old_lkl != FLT_MAX) {
			if (abs(old_lkl - new_lkl) < 1e-3)
				return;
		}
		old_lkl = new_lkl;
		//system("ECHO iter");
	}

}

void Gaussian_Mixture_Model::Classify(std::list<Eigen::VectorXf>* label_density, std::list<Eigen::VectorXf>& X) {

	label_density->clear();
	for (auto it = X.begin(); it != X.end(); it++) {
		label_density->push_back(VectorXf());
		this->Classify(&label_density->back(), *it);
	}

}

void Gaussian_Mixture_Model::get_parameters(std::list<float>* weights, std::list<Eigen::VectorXf>* Means, std::list<Eigen::MatrixXf>* Covariances) {

	weights->clear();
	Means->clear();
	Covariances->clear();

	for (auto it = this->Clusters.begin(); it != this->Clusters.end(); it++) {
		weights->push_back(it->weight);
		Means->push_back(it->Mean);
		Covariances->push_back(it->Covariance);
	}

}

void Gaussian_Mixture_Model::__check_eig_Cov() {

	size_t k, K = this->get_Feature_size();
	for (auto it = this->Clusters.begin(); it != this->Clusters.end(); it++) {
		EigenSolver<MatrixXf> eig_solv(it->Covariance);
		auto eigs = eig_solv.eigenvalues();
		
	//check all values are sufficient high
		for (k = 0; k < K; k++) {
			if ( abs (eigs(k).real()) < 0.0001f ) {
				system("ECHO warning: detected at least one cluster with a too low covariance");
				return;
			}
		}
	}

}

Gaussian_Mixture_Model::Gaussian_Mixture_Model(const size_t& N_clusters, const size_t& dimension_size) {

	if (N_clusters == 0) abort();
	if (dimension_size == 0) abort();

	size_t N_sample = 100 * N_clusters;

	list<VectorXf> samples;
	size_t kk;
	for (size_t k = 0; k < N_sample; k++) {
		samples.push_back(VectorXf(dimension_size));
		for (kk = 0; kk < dimension_size; kk++)
			samples.back()(kk) = 2.f * (float)rand() / (float)RAND_MAX - 1.f;
	}

	Train_set temp(samples);

	this->__EM_train(temp, N_clusters, NULL);

}

void Gaussian_Mixture_Model::__copy(const Gaussian_Mixture_Model& to_clone) {

	this->Clusters.clear();
	for (auto it = to_clone.Clusters.begin(); it != to_clone.Clusters.end(); it++)
		this->Clusters.push_back(*it);

}

Gaussian_Mixture_Model::Gaussian_Mixture_Model(const Train_set& train_set, const std::list<size_t>& N_clusters_to_try, std::list<std::list<float>>* likelihood_story) {

	if (N_clusters_to_try.empty()) abort();

	struct GMM_info {
		Gaussian_Mixture_Model*		model;
		list<float>					log_lkl;
	};
	list<GMM_info> trials;

	list<list<float>> temp;
	for (auto it = N_clusters_to_try.begin(); it != N_clusters_to_try.end(); it++) {
		trials.push_back(GMM_info());
		trials.back().model = new Gaussian_Mixture_Model(train_set, *it, &temp);
		trials.back().log_lkl = temp.front();
	}

	GMM_info* best_fitting = &trials.front();
	auto it = trials.begin(); it++;
	for (it; it != trials.end(); it++) {
		if (it->log_lkl.back() > best_fitting->log_lkl.back())
			best_fitting = &(*it);
	}

	this->__copy(*best_fitting->model);

	if (likelihood_story != NULL) {
		likelihood_story->clear();
		for (it = trials.begin(); it != trials.end(); it++)
			likelihood_story->push_back(it->log_lkl);
	}

	for (it = trials.begin(); it != trials.end(); it++)
		delete it->model;

	this->__check_eig_Cov();

}

Gaussian_Mixture_Model::Gaussian_Mixture_Model(const Train_set& train_set, const size_t& N_clusters, const bool& do_trials, std::list<std::list<float>>* likelihood_story) {

	if (do_trials) {
		if (N_clusters == 0) abort();

		list<size_t> trials;
		for (size_t k = 0; k < N_clusters; k++)
			trials.push_back(k);

		auto twin_temp = new Gaussian_Mixture_Model(train_set, trials, likelihood_story);
		this->__copy(*twin_temp);
		delete twin_temp;

	}
	else {
		if (likelihood_story == NULL)
			this->__EM_train(train_set, N_clusters, NULL);
		else {
			likelihood_story->clear();
			likelihood_story->push_back(list<float>());
			this->__EM_train(train_set, N_clusters, &likelihood_story->front());
		}
	}

	this->__check_eig_Cov();

}

Gaussian_Mixture_Model::Gaussian_Mixture_Model(const std::list<float>& weights, const  std::list<Eigen::VectorXf>& Means, const std::list<Eigen::MatrixXf>& Covariances) {

	if (weights.empty()) abort();
	if (weights.size() != Means.size()) abort();
	if (weights.size() != Covariances.size()) abort();

	size_t n = Means.front().size();
	if (n == 0) abort();

	list<float> w(weights);
	normalize_list(w);

	auto it_M = Means.begin();
	auto it_C = Covariances.begin();
	for (auto it_w = w.begin(); it_w != w.end(); it_w) {

		if (*it_w <= 0.f) abort();
		if (it_M->size() != n) abort();
		if (it_C->rows() != n) abort();
		if (it_C->cols() != n) abort();

		this->Clusters.push_back(cluster(*it_w, *it_M, *it_C));

		it_M++;
		it_C++;
	}

	this->__check_eig_Cov();

}

void Gaussian_Mixture_Model::get_parameters(Eigen::MatrixXf* packed_params) {

	size_t N_cluster = this->Clusters.size();
	size_t n = this->Clusters.front().Mean.size();

	*packed_params = MatrixXf(n*N_cluster, 2 + n);
	packed_params->setZero();
	size_t line = 0;
	for (auto it = this->Clusters.begin(); it != this->Clusters.end(); it++) {
		(*packed_params)(line , 0) = it->weight;
		(*packed_params).block(line , 1, n, 1) = it->Mean;
		(*packed_params).block(line, 2, n, n) = it->Covariance;
		
		line += n;
	}


}

Gaussian_Mixture_Model::Gaussian_Mixture_Model(Eigen::MatrixXf& packed_params) {

	size_t n = packed_params.cols() - 2;
	if (n == 0) abort();

	if ((packed_params.rows() % n) != 0) abort();

	size_t N_cluster = (size_t)((float)packed_params.rows() / (float)n);

	VectorXf M_temp;
	MatrixXf C_temp;
	size_t line = 0;
	for (size_t k = 0; k < N_cluster; k++) {
		if (packed_params(line, 0) < 0.f) abort();

		M_temp = packed_params.block(line, 1, n, 1);
		C_temp = packed_params.block(line, 2, n, n);

		this->Clusters.push_back(cluster(packed_params(line, 0), M_temp, C_temp));

		line += n;
	}

	this->__check_eig_Cov();

}