#include <list>
#include <fstream>
#include <Eigen\Dense>
using namespace std;
using namespace Eigen;


//sample random points in the hypercube centred in the origin with sizes equal to Sizes
void get_random_samples(list<VectorXf>* samples, VectorXf& Sizes, const size_t& N_sample) {

	if (Sizes.size() == 0) abort();

	VectorXf A, B;
	A = Sizes;
	size_t kk;
	for (kk = 0; kk < (size_t)Sizes.size(); kk++)
		A(kk) = abs(Sizes(kk));
	B = 0.5f* A;

	samples->clear();
	for (size_t k = 0; k < N_sample; k++) {
		samples->push_back(VectorXf(Sizes.size()));

		for (kk = 0; kk < (size_t)Sizes.size(); kk++) {
			samples->back()(kk) = A(kk)* (float)rand() / (float)RAND_MAX - B(kk);
		}
	}

}

//plot row wise the vector in the passed list
void print_Vectors(const string& file_name, list<VectorXf>& Samples)
 {

	ofstream f(file_name);
	if (!f.is_open())
		abort();

	auto it = Samples.begin();
	while (it != Samples.end()) {
		f << it->transpose();
		it++;
		if (it != Samples.end())
			f << endl;
	}

	f.close();

}

//plot row wise the clusters: first number of the row is the cluster ID, then row wise all the values of the vector
void print_clusters(const string& file_name, list<list<VectorXf*>>& Samples) {

	ofstream f(file_name);
	if (!f.is_open())
		abort();

	list<VectorXf*>::iterator it2;
	size_t k = 0;
	for (auto it = Samples.begin(); it != Samples.end(); it++) {
		for (it2 = it->begin(); it2 != it->end(); it2++)
			f << k << " " << (*it2)->transpose() << endl;
		k++;
	}

	f.close();

}

void print_GMM_params(const string& file_name, std::list<float>& weights, std::list<Eigen::VectorXf>& Means, std::list<Eigen::MatrixXf>& Covariances) {

	ofstream f(file_name);
	if (!f.is_open())
		abort();

	auto it_w = weights.begin();
	auto it_M = Means.begin();
	auto it_C = Covariances.begin();

	size_t n = it_M->size(), r;

	for (it_w; it_w != weights.end(); it_w++) {

		for (r = 0; r < n; r++) {
			if (r == 0) f << *it_w << " ";
			else f << 0.f << " ";

			f << (*it_M)(r) << " ";
			f << it_C->row(r);
			f << endl;
		}

		it_M++;
		it_C++;
	}

	f.close();

}