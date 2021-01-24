/**
 * Author:    Andrea Casalino
 * Created:   03.12.2019
*
* report any bug to andrecasa91@gmail.com.
 **/

#include "../GMM/GMM.h"
#include <iostream>
#include <fstream>
using namespace std;
using namespace Eigen;

int main() {

//sample a random 2d GMM model
	size_t N_clusters = 5;
	Gaussian_Mixture_Model random_model( N_clusters, 2);

//get samples from the random model
	list<VectorXf>  train_set;
	size_t	train_set_size = 500;
	random_model.Get_samples(&train_set, train_set_size);

//fit a model considering the sampled train set
	Gaussian_Mixture_Model::Train_set set(train_set);
	Gaussian_Mixture_Model learnt_model(N_clusters, set);

//log the two models to visually check the differences
	string params;
	params = random_model.get_paramaters_as_JSON();
	ofstream f_random("random_model.json");
	f_random << params;
	f_random.close();
	params = learnt_model.get_paramaters_as_JSON();
	ofstream f_learnt("learnt_model.json");
	f_learnt << params;
	f_learnt.close();
	
	//use the python script Visualize.py to see the results

	return 0;
}
