/**
 * Author:    Andrea Casalino
 * Created:   03.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianMixtureModel/GaussianMixtureModel.h>
#include <GaussianMixtureModel/GaussianMixtureModelFactory.h>
#include <GaussianMixtureModel/ExpectationMaximization.h>
#include "Utils.h"

/////////////////////////
// GMM in 2 dimensions //
/////////////////////////
int main() {
//sample a random 2d GMM model
	size_t N_clusters = 5;
	auto random_model = gauss::gmm::GaussianMixtureModelFactory(2, N_clusters).makeRandomModel();

//get samples from the random model
	gauss::TrainSet train_set(random_model->drawSamples(500));

//fit a model using expectation maximization, considering the sampled train set
	gauss::gmm::GaussianMixtureModel learnt_model(gauss::gmm::ExpectationMaximization(train_set, N_clusters));

//log the two models to visually check the differences
	print(*random_model, "random_model2d.json");
	print(learnt_model, "learnt_model2d.json");
	
// use the python script Visualize02.py to see the results

	return EXIT_SUCCESS;
}
