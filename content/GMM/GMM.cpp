/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
*
* report any bug to andrecasa91@gmail.com.
 **/

#include "GMM.h"
#include <fstream>
#include <string>
#include <sstream>
#include <random>
using namespace std;

#include <Eigen\Cholesky>
using namespace Eigen;



#define log_2_pi logf(2.f * 3.141592f)
#define PI_GREEK 3.14159f
default_random_engine generator;




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

	if (slices.empty()) throw INVALID_TRAINING_SET;

	*V = VectorXf(slices.size());
	auto it = slices.begin();
	for (size_t k = 0; k < slices.size(); k++) {
		(*V)(k) = (float)atof(it->c_str());
		it++;
	}

}

Gaussian_Mixture_Model::Train_set::Train_set(const std::string& file_to_read)  {

	ifstream f(file_to_read);

	if (!f.is_open()) throw INVALID_TRAINING_SET;

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

	this->__append_samples(raw);

	if (this->Samples.empty()) throw INVALID_TRAINING_SET;

}

Gaussian_Mixture_Model::Train_set::Train_set(const std::list<Eigen::VectorXf>& samples)  {

	this->__append_samples(samples); 
	if (this->Samples.empty()) throw INVALID_TRAINING_SET;

};

Gaussian_Mixture_Model::Train_set::Train_set(const std::vector<Eigen::VectorXf>& samples) {

	this->__append_samples(samples);
	if (this->Samples.empty()) throw INVALID_TRAINING_SET;

}

void Gaussian_Mixture_Model::Train_set::Set_initial_guess(const std::list<size_t>& cluster_initial_guess) {

	if (cluster_initial_guess.size() != this->Samples.size()) throw INVALID_INITIAL_GUESS;

	this->initial_guess_clusters.clear();
	auto it = cluster_initial_guess.begin();
	size_t N_clusters = *it;
	it++;
	for (it; it != cluster_initial_guess.end(); it++) {
		if (*it > N_clusters) N_clusters = *it;
	}

	for (size_t k = 0; k <= N_clusters; k++) 
		this->initial_guess_clusters.push_back(list<const VectorXf*>());
	auto it_cl = this->initial_guess_clusters.begin();

	auto it_S = this->Samples.begin();
	for (it = cluster_initial_guess.begin(); it != cluster_initial_guess.end(); it++) {
		it_cl = this->initial_guess_clusters.begin();
		advance(it_cl, *it);
		it_cl->push_back(&(*it_S));
		it_S++;
	}

	for (it_cl = this->initial_guess_clusters.begin(); it_cl != initial_guess_clusters.end(); it_cl++) {
		if (it_cl->empty()) throw INVALID_INITIAL_GUESS;
	}

}




void get_mean(VectorXf* Mean, list<const VectorXf*>* l) {

	if (l->empty()) abort(); //report this bug to andrecasa91@gmail.com

	auto it = l->begin();
	*Mean = *l->front();
	it++;
	for (it; it != l->end(); it++)
		*Mean += *(*it);
	*Mean = (1.f / (float)l->size()) * (*Mean);

}

void recompute_means(list<VectorXf>* Means, list<list<const VectorXf*>>& clusters) {

	Means->clear();
	for (auto it = clusters.begin(); it != clusters.end(); it++) {
		Means->push_back(VectorXf());
		get_mean(&Means->back(), &(*it));
	}

}

bool check_are_the_same(const list<list<const VectorXf*>>& a, const list<list<const VectorXf*>>& b) {

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

void Forgy_init(list<VectorXf>* Means, const Gaussian_Mixture_Model::Train_set& Set, const size_t& N_means) {

	auto Samples = Set.Get_Samples();
	list<const VectorXf*> samples;
	for (auto it = Samples->begin(); it != Samples->end(); it++) 
		samples.push_back(&(*it));

	size_t pos_rand;
	auto it_s = samples.begin();
	for (size_t k = 0; k < N_means; k++) {
		pos_rand = rand() % samples.size();
		it_s = samples.begin();
		advance(it_s, pos_rand);
		Means->push_back(**it_s);
		it_s = samples.erase(it_s);
	}

};

void Gaussian_Mixture_Model::K_means_clustering(std::list<std::list<const Eigen::VectorXf*>>* clusters, const Train_set& Set, const size_t& N_means, const size_t& Iterations) {

	auto Samples = Set.Get_Samples();

	if (N_means == 0) throw INVALID_NUMBER_OF_CLUSTERS;
	if (N_means > Samples->size()) throw INVALID_NUMBER_OF_CLUSTERS;

	list<VectorXf> Means;
	auto it_s = Samples->begin();
	auto it = Means.begin();

	clusters->clear();
	for (size_t k = 0; k < N_means; k++)
		clusters->push_back(list<const VectorXf*>());
	Forgy_init(&Means , Set, N_means);

	size_t Iter = Iterations;
	if (Iter < N_means) Iter = N_means;

	auto it_c = clusters->begin();
	float dist_min, temp;
	size_t pos_nearest, kk;
	list<list<const VectorXf*>> old_clustering;
	for (size_t k = 0; k < Iter; k++) {
		for (it_c = clusters->begin(); it_c != clusters->end(); it_c++)
			it_c->clear();

		for (it_s = Samples->begin(); it_s != Samples->end(); it_s++) {
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
	}

}




void invert_symm_positive(MatrixXf* Sigma_inverse, MatrixXf& Sigma) {

	//LLT<MatrixXf> lltOfCov(Sigma);
	//MatrixXf L(lltOfCov.matrixL());
	//*Sigma_inverse = L * L.transpose();

	*Sigma_inverse = Sigma.inverse();

}

void eval_Normal_Log_density(float* den, const VectorXf& Mean, const MatrixXf& invCov, const float& absDetCov, const VectorXf& X) {

	*den = (X - Mean).transpose() * invCov * (X - Mean);
	*den += X.size() * log_2_pi;
	*den += logf(absDetCov);

	*den *= -0.5f;

}

float Gaussian_Mixture_Model::__EM_train(const Train_set&   train_set, const size_t& N_clusters, const size_t& Iterations) {

	if (N_clusters == 0) throw INVALID_NUMBER_OF_CLUSTERS;

	auto Samples = train_set.Get_Samples();

	auto it_s = Samples->begin();
	if (this->Clusters.size() != N_clusters) {
		//initialize clusters with k-means
		list<list<const VectorXf*>> clst;

		auto initial_guess = train_set.Get_guess();

		if ((!initial_guess.empty()) && (initial_guess.size() == N_clusters)) clst = initial_guess;
		else this->K_means_clustering(&clst, train_set, N_clusters, Iterations);

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

			this->__append_cluster(1.f / (float)clst.size(), M_temp, C_temp);
		}
	}

	//EM loop
	size_t Iter = Iterations;
	if (Iter < N_clusters) Iter = N_clusters;
	int R = (int)Samples->size();
	int C = (int)this->Clusters.size();
	auto it_c = this->Clusters.begin();
	MatrixXf gamma(R, C);
	int r, c;
	float temp;
	VectorXf n(C);
	float old_lkl = FLT_MAX, new_lkl;
	for (int k = 0; k < Iter; k++) {
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
				it_c->Covariance += gamma(r, c) * (*it_s - it_c->Mean) * (*it_s - it_c->Mean).transpose();
				r++;
			}
			it_c->Covariance *= 1.f / n(c);
			c++;

			invert_symm_positive(&it_c->Inverse_Cov, it_c->Covariance);
			it_c->Abs_Deter_Cov = abs(it_c->Covariance.determinant());
		}

		new_lkl = 0.f;
		for (it_s = Samples->begin(); it_s != Samples->end(); it_s++) {
			temp = this->__eval_log_density(*it_s);
			new_lkl += temp;
		}

		if (old_lkl != FLT_MAX) {
			if (abs(old_lkl - new_lkl) < 1e-3)
				return new_lkl;
		}
		old_lkl = new_lkl;
	}
	return old_lkl;

}

void Gaussian_Mixture_Model::__append_cluster(const float& w, const Eigen::VectorXf& M, const Eigen::MatrixXf& C)  {

	this->Clusters.push_back(cluster());
	this->Clusters.back().weight = w;
	this->Clusters.back().Mean = M;
	this->Clusters.back().Covariance = C;

	invert_symm_positive(&this->Clusters.back().Inverse_Cov, this->Clusters.back().Covariance);
	this->Clusters.back().Abs_Deter_Cov = abs(this->Clusters.back().Covariance.determinant());

}

float Gaussian_Mixture_Model::__eval_log_density(const VectorXf& X) const {

	float den = 0.f;
	float temp;
	for (auto it_cl = this->Clusters.begin(); it_cl != this->Clusters.end(); it_cl++) {
		eval_Normal_Log_density(&temp, it_cl->Mean, it_cl->Inverse_Cov, it_cl->Abs_Deter_Cov, X);
		den += expf(logf(it_cl->weight) + temp);
	}
	den = logf(den);
	return den;

};

Gaussian_Mixture_Model::Gaussian_Mixture_Model(const size_t& N_clusters, const Train_set& train_set, const size_t& Iterations, float* train_set_lklhood ) {

	float temp = this->__EM_train(train_set, N_clusters, Iterations);
	if (train_set_lklhood != NULL) *train_set_lklhood = temp;

}

Gaussian_Mixture_Model::Gaussian_Mixture_Model(const size_t& N_clusters, const size_t& dimension_size) {

	if (N_clusters == 0) throw INVALID_NUMBER_OF_CLUSTERS;
	if (dimension_size == 0) throw INVALID_INPUT;

	size_t N_sample = 100 * N_clusters;

	list<VectorXf> samples;
	size_t kk;
	for (size_t k = 0; k < N_sample; k++) {
		samples.push_back(VectorXf(dimension_size));
		for (kk = 0; kk < dimension_size; kk++)
			samples.back()(kk) = 2.f * (float)rand() / (float)RAND_MAX - 1.f;
	}

	Train_set temp(samples);

	this->__EM_train(temp, N_clusters, 100 * samples.front().size());

}

void check_consistency(const MatrixXf Cov) {

	size_t K = (size_t)Cov.cols();
	size_t c;
	for (size_t r = 0; r < K; r++) {
		for (c = (r + 1); c < K; c++) {
			if (abs(Cov(r, c) - Cov(c, r)) > 1e-5) throw INVALID_GMM_PARAMETERS;
		}
	}

	EigenSolver<MatrixXf> eig_solv(Cov);
	auto eigs = eig_solv.eigenvalues();

	//check all values are sufficient high
	for (size_t k = 0; k < K; k++) {
		if (abs(eigs(k).real()) < 1e-5) throw INVALID_GMM_PARAMETERS;
	}

};

Gaussian_Mixture_Model::Gaussian_Mixture_Model(const Eigen::MatrixXf& params) {

	size_t x = (size_t)params.cols();
	size_t Ncl = (size_t)params.rows() / (x+2);
	if (Ncl * (x + 2) != params.rows()) throw INVALID_GMM_PARAMETERS;

	size_t r;
	VectorXf Mean(x);
	MatrixXf Cov(x, x);
	float w;
	for (size_t k = 0; k < Ncl; k++) {
		w = params((2 + x)*k, 0);
		if (w < 0) throw INVALID_GMM_PARAMETERS;
		if (w > 1.f) throw INVALID_GMM_PARAMETERS;

		Mean = params.row((2 + x)*k +1).transpose();
		for (r = 0; r < x; r++)
			Cov.row(r) = params.row((2 + x)*k + r + 2);
		check_consistency(Cov);

		this->__append_cluster(w, Mean, Cov);
	}

}

Gaussian_Mixture_Model Gaussian_Mixture_Model::Fit_optimal_model(const Train_set& train_set, const std::list<size_t>& N_clusters_to_try, const size_t& Iterations, float* train_set_lklhood) {

	if (N_clusters_to_try.empty()) throw INVALID_INPUT;
	
	struct model {
		Gaussian_Mixture_Model*  model;
		float												lkl;
	};
	list<model> models;
	for (auto it = N_clusters_to_try.begin(); it != N_clusters_to_try.end(); it++) {
		models.push_back(model());
		models.back().model = new Gaussian_Mixture_Model(*it, train_set, Iterations, &models.back().lkl);
	}

	auto it = models.begin();
	model* best_model = &(*it);
	it++;
	for (it; it != models.end(); it++) {
		if (it->lkl > best_model->lkl)
			best_model = &(*it);
	}

	if (train_set_lklhood != NULL) *train_set_lklhood = best_model->lkl;
	Gaussian_Mixture_Model temp(*best_model->model);

	for (it = models.begin(); it != models.end(); it++) delete it->model;

	return temp;

}

void Gaussian_Mixture_Model::Classify(VectorXf* label_density, VectorXf& X) const {

	*label_density = VectorXf(this->Clusters.size());
	size_t k = 0;
	float temp;
	for (auto it = this->Clusters.begin(); it != this->Clusters.end(); it++) {
		eval_Normal_Log_density(&temp, it->Mean, it->Inverse_Cov, it->Abs_Deter_Cov, X);
		(*label_density)(k) = expf(logf(it->weight) + temp);
		k++;
	}

	*label_density *= (1.f / label_density->sum());

}

void Gaussian_Mixture_Model::Classify(std::list<Eigen::VectorXf>* label_density, std::list<Eigen::VectorXf>& X) const {

	label_density->clear();
	for (auto it = X.begin(); it != X.end(); it++) {
		label_density->push_back(VectorXf());
		this->Classify(&label_density->back(), *it);
	}

}

class Gaussian_sampler {
public:
	Gaussian_sampler(const Eigen::VectorXf& Mean, const Eigen::MatrixXf& Sigma) : gauss_iso(0.f, 1.f) {

		this->Trasl = Mean;
		LLT<MatrixXf> lltOfCov(Sigma);
		this->Rot = lltOfCov.matrixL();

	}

	void get_sample(Eigen::VectorXf* sample) {

		*sample = VectorXf(this->Trasl.size());
		for (size_t k = 0; k < (size_t)this->Trasl.size(); k++)
			(*sample)(k) = gauss_iso(generator);

		*sample = this->Rot * *sample;
		*sample += this->Trasl;

	}
private:
	Eigen::VectorXf						Trasl;
	Eigen::MatrixXf						Rot;
	std::normal_distribution<float>		    gauss_iso;
};

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

void Gaussian_Mixture_Model::Get_samples(std::list<Eigen::VectorXf>* samples, const size_t& Number_of_samples) const {

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
	for (size_t k = 0; k < Number_of_samples; k++) {
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

Eigen::MatrixXf Gaussian_Mixture_Model::get_parameters_as_matrix() const {

	size_t D = 2 + this->Clusters.front().Mean.size();
	MatrixXf M(this->Clusters.size()* D, this->Clusters.front().Mean.size());

	size_t r, R = this->Clusters.front().Mean.size();
	for (size_t k = 0; k < this->Clusters.size(); k++) {
		M.row(D*k).setZero();
		M.row(D*k)(0) = this->Clusters[k].weight;
		M.row(D*k + 1) = this->Clusters[k].Mean.transpose();

		for (r = 0; r < R; r++) 
			M.row(D*k +  r + 2) = this->Clusters[k].Covariance.row(r);
	}

	return M;

}

void print_vector(string* S, const VectorXf& V) {

	*S += "[";
	*S += to_string(V(0));
	for (size_t k = 1;  k < (size_t)V.size(); k++) {
		*S += ",";
		*S += to_string(V(k));
	}
	*S += "]";

}
void append_cluster(string* S, const Gaussian_Mixture_Model::cluster& cl) {

	*S += "\n{";

	*S += "\"w\":" + to_string(cl.weight) + ",\n";

	*S += "\"Mean\":" ;
	print_vector(S, cl.Mean);
	*S += ",\n";

	*S += "\"Covariance\":";
	*S += "[";
	print_vector(S, cl.Covariance.row(0));
	for (size_t k = 1; k < (size_t)cl.Covariance.rows(); k++) {
		*S += ",";
		print_vector(S, cl.Covariance.row(k));
	}
	*S += "]\n";

	*S += "}\n";
}
string Gaussian_Mixture_Model::get_paramaters_as_JSON() const {

	string param ="[";
	auto it = this->Clusters.begin();
	append_cluster(&param, *it);
	it++;
	for (it; it != this->Clusters.end(); it++) {
		param += ",";
		append_cluster(&param, *it);
	}
	param += "]";

	return param;

}

float Gaussian_Mixture_Model::Get_KULLBACK_LEIBLER_divergence_MonteCarlo(const Gaussian_Mixture_Model& other) const {

	if (this->get_Space_size() != other.get_Space_size()) throw INVALID_INPUT;

	list<VectorXf> Samples;
	size_t N_samples = (size_t)this->Clusters.front().Mean.size() * 500;
	this->Get_samples( &Samples, N_samples);

	float lkl_A, lkl_B;
	float div = 0.f;
	for (auto it = Samples.begin(); it != Samples.end(); it++) {
		lkl_A = this->__eval_log_density(*it);
		lkl_B = other.__eval_log_density(*it);

		div += lkl_A;
		div -= lkl_B;
	}
	div *= 1.f / (float)N_samples;
	return div;

}

float Divergence_Normals(const Gaussian_Mixture_Model::cluster& f, const Gaussian_Mixture_Model::cluster& g) {

	float temp = logf(g.Abs_Deter_Cov) - logf(f.Abs_Deter_Cov);
	MatrixXf P = g.Inverse_Cov * f.Covariance;
	temp += P.trace();
	VectorXf Delta = f.Mean - g.Mean;
	temp += Delta.transpose() * g.Inverse_Cov * Delta;
	temp -= (float)f.Mean.size();
	temp *= 0.5f;
	return temp;

}

float t_operator(const Gaussian_Mixture_Model::cluster& f, const Gaussian_Mixture_Model::cluster& g) {

	float temp = -f.Mean.size()*logf(2.f * PI_GREEK);
	MatrixXf S = f.Covariance;
	S += g.Covariance;
	temp -= logf(S.determinant());
	VectorXf Delta = g.Mean - f.Mean;
	temp -= Delta.transpose() * S.inverse() * Delta;
	temp *= 0.5f;
	return expf(temp);

}

void Gaussian_Mixture_Model::Get_KULLBACK_LEIBLER_divergence_estimate(const Gaussian_Mixture_Model& other, float* upper_bound, float* lower_bound) const {

	if (this->get_Space_size() != other.get_Space_size()) throw INVALID_INPUT;

	MatrixXf Divergences(this->Clusters.size() , other.Clusters.size());
	MatrixXf t(this->Clusters.size(), other.Clusters.size());
	MatrixXf z(this->Clusters.size(), this->Clusters.size());
	size_t a, A = this->Clusters.size(), b, B = other.Clusters.size();
	for (a = 0; a < A; a++) {
		for (b = 0; b < B; b++) {
			Divergences(a, b) = Divergence_Normals(this->Clusters[a], other.Clusters[b]);
			t(a,b) = t_operator(this->Clusters[a], other.Clusters[b]);
		}
	}
	for (a = 0; a < A; a++) {
		for (b = 0; b < A; b++) {
			z(a, b) = t_operator(this->Clusters[a], this->Clusters[b]);
		}
	}

	*upper_bound = 0.f;
	*lower_bound = 0.f;
	size_t a2;

	float temp = 0.f, temp2, temp3;
	for (a = 0; a < A; a++) {
		temp2 = 0.f;
		for (a2 = 0; a2 < A; a2++)  temp2 += this->Clusters[a2].weight * z(a, a2);
		temp2 = logf(temp2);
		temp3 = 0.f;
		for (b = 0; b < B; b++)  temp3 += other.Clusters[b].weight * expf(-Divergences(a, b));
		temp3 = logf(temp3);
		*upper_bound += this->Clusters[a].weight * (temp2 - temp3);

		temp2 = 0.f;
		for (a2 = 0; a2 < A; a2++)  temp2 += this->Clusters[a2].weight * expf(-Divergences(a, b));
		temp2 = logf(temp2);
		temp3 = 0.f;
		for (b = 0; b < B; b++)  temp3 += other.Clusters[b].weight * t(a, b);
		temp3 = logf(temp3);
		*lower_bound += this->Clusters[a].weight * (temp2 - temp3);

		temp += this->Clusters[a].weight * 0.5f * logf(powf(2.f* PI_GREEK * 2.71828f, (float)this->Clusters[a].Mean.size()) * this->Clusters[a].Abs_Deter_Cov);
	}

	*upper_bound += temp;
	*lower_bound -= temp;

}