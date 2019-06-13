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





void print_Vectors(ostream& f, list<VectorXf>& Samples)
{

	auto it = Samples.begin();
	while (it != Samples.end()) {
		f << it->transpose();
		it++;
		if (it != Samples.end())
			f << endl;
	}

}

void print_clusters(ostream& f, list<list<VectorXf*>>& Samples) {

	list<VectorXf*>::iterator it2;
	size_t k = 0;
	for (auto it = Samples.begin(); it != Samples.end(); it++) {
		for (it2 = it->begin(); it2 != it->end(); it2++)
			f << k << " " << (*it2)->transpose() << endl;
		k++;
	}

}

void print_Matrix(ostream& f, Eigen::MatrixXf& Mat) {

	size_t r, c, R = Mat.rows(), C = Mat.cols();
	for (r = 0; r < R; r++) {
		for (c = 0; c < C; c++) {
			f << " " << Mat(r, c);
		}
		f << endl;
	}

}



//plot row wise the vector in the passed list
void print_Vectors(const string& file_name, list<VectorXf>& Samples)
 {

	ofstream f(file_name);
	if (!f.is_open())
		abort();

	print_Vectors(f, Samples);

	f.close();

}

//plot row wise the clusters: first number of the row is the cluster ID, then row wise all the values of the vector
void print_clusters(const string& file_name, list<list<VectorXf*>>& Samples) {

	ofstream f(file_name);
	if (!f.is_open())
		abort();

	print_clusters(f, Samples);

	f.close();

}

void print_Matrix(const string& file_name, Eigen::MatrixXf& Mat) {

	ofstream f(file_name);
	if (!f.is_open())
		abort();

	print_Matrix(f, Mat);

	f.close();

}
