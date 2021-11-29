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
// GMM in 3 dimensions //
/////////////////////////
int main() {
//sample a random 3d GMM model
	size_t N_clusters = 6;
	auto random_model = gauss::gmm::GaussianMixtureModelFactory(3, N_clusters).makeRandomModel();

//get samples from the random model
	gauss::TrainSet train_set(random_model->drawSamples(1000));

//fit a model using expectation maximization, considering the sampled train set
	gauss::gmm::GaussianMixtureModel learnt_model(gauss::gmm::ExpectationMaximization(train_set, N_clusters));

//log the two models to visually check the differences
	print(*random_model, "random_model3d.json");
	print(learnt_model, "learnt_model3d.json");

//use the python script Visualize03.py to see the results

	return EXIT_SUCCESS;
}