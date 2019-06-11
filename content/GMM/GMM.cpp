//author: Andrea Casalino
//mail:andrecasa91@gmail.com

#include "GMM.h"
#include <fstream>
#include <string>
#include <sstream>
#include <random>
using namespace std;
#include <Eigen\Cholesky>
using namespace Eigen;

#define log_2_pi logf(2.f * 3.141592f);
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
	this->Samples.push_back(*it);
	it++;
	for (it; it != s.end(); it++) {
		if (it->size() != D) 
			abort();		
		this->Samples.push_back(*it);
	}

};

Gaussian_Mixture_Model::Train_set::Train_set(const std::string& file_to_read) {

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







void eval_Normal_Log_density(float* den, VectorXf& Mean, MatrixXf& invCov, float& absDetCov, VectorXf& X) {

	*den = (X - Mean).transpose() * invCov * (X - Mean);
	*den += X.size() * log_2_pi;
	*den += logf(absDetCov);

	*den *= -0.5f;

}

Gaussian_Mixture_Model::Gaussian_Mixture_Model(const Gaussian_Mixture_Model& to_clone) {

	for (auto it = to_clone.Clusters.begin(); it != to_clone.Clusters.end(); it++) {
		this->Clusters.push_back(cluster());
		this->Clusters.back().Mean			= it->Mean;
		this->Clusters.back().Covariance	= it->Covariance;
		this->Clusters.back().Inverse_Cov	= it->Inverse_Cov;
		this->Clusters.back().Abs_Deter_Cov = it->Abs_Deter_Cov;
		this->Clusters.back().weight		= it->weight;
	}

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

	*label_density = (1.f / label_density->sum()) * (*label_density);

}

class Gaussian_sampler {
public:
	Gaussian_sampler(VectorXf* Mean, MatrixXf& Sigma);

	void get_sample(VectorXf* sample);
private:
	VectorXf*						Trasl;
	MatrixXf						Rot;
	normal_distribution<float>		gauss_iso;
};

Gaussian_sampler::Gaussian_sampler(VectorXf* Mean, MatrixXf& Sigma) : gauss_iso(0.f, 1.f) {

	this->Trasl = Mean;
	LLT<MatrixXf> lltOfCov(Sigma);
	this->Rot = lltOfCov.matrixL();

}

void Gaussian_sampler::get_sample(VectorXf* sample) {

	*sample = VectorXf(this->Trasl->size());
	for (size_t k = 0; k < (size_t)this->Trasl->size(); k++)
		(*sample)(k) = gauss_iso(generator);

	*sample = this->Rot * *sample;
	*sample += *this->Trasl;

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
		Samplers.push_back(new Gaussian_sampler(&it->Mean, it->Covariance));
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

void Gaussian_Mixture_Model::EM_train(Train_set&   train_set, const size_t& N_clusters, std::list<float>* likelihood_story) {

	if (N_clusters == 0) abort(); //TODO
	else {
		this->__EM_train(train_set, N_clusters, likelihood_story);
	}

}

struct Gaussian_Mixture_Model::Train_set::Sample_handler {
	static std::list<Eigen::VectorXf>* get_samples(Gaussian_Mixture_Model::Train_set& set) { return &set.Samples; };
};

void Gaussian_Mixture_Model::__EM_train(Train_set&   train_set, const size_t& N_clusters, list<float>* likelihood_story) {

	if (likelihood_story != NULL)
		likelihood_story->clear();

	auto sample_raw = Gaussian_Mixture_Model::Train_set::Sample_handler::get_samples(train_set);
	list<VectorXf>* Samples = Gaussian_Mixture_Model::Train_set::Sample_handler::get_samples(train_set);

	auto it_s = Samples->begin();
	if (this->Clusters.size() != N_clusters) {
	//initialize clusters with k-means
		list<list<VectorXf*>> clst;

		K_means::do_clustering(&clst, *Samples, N_clusters);
		this->Clusters.clear();
		for (auto it = clst.begin(); it != clst.end(); it++) {
			this->Clusters.push_back(cluster());
			
			get_mean(&this->Clusters.back().Mean, &(*it));
			this->Clusters.back().Covariance = MatrixXf(this->Clusters.back().Mean.size(), this->Clusters.back().Mean.size());
			this->Clusters.back().Covariance.setZero();
			for (it_s = Samples->begin(); it_s != Samples->end(); it_s++) {
				this->Clusters.back().Covariance += (*it_s - this->Clusters.back().Mean) * (*it_s - this->Clusters.back().Mean).transpose();
			}
			this->Clusters.back().Covariance *= 1.f / (float)Samples->size();

			this->Clusters.back().weight = 1.f / (float)this->Clusters.size();
			this->Clusters.back().Inverse_Cov = this->Clusters.back().Covariance.inverse();
			this->Clusters.back().Abs_Deter_Cov = abs(this->Clusters.back().Covariance.determinant());
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

			it_c->Inverse_Cov = it_c->Covariance.inverse();
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