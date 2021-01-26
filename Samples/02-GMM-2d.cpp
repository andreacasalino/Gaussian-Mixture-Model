/**
 * Author:    Andrea Casalino
 * Created:   03.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "Utils.h"

/////////////////////////
// GMM in 2 dimensions //
/////////////////////////
int main() {
//sample a random 2d GMM model
	size_t N_clusters = 5;
	gmm::GMM random_model( N_clusters, 2);

//get samples from the random model
	list<gmm::V> train_set = random_model.drawSamples(500);

//fit a model considering the sampled train set
	gmm::GMM learnt_model(N_clusters, gmm::TrainSet(train_set));

//log the two models to visually check the differences
	print(random_model, "random_model2d.json");
	print(learnt_model, "learnt_model2d.json");
	
// use the python script Visualize02.py to see the results

	return EXIT_SUCCESS;
}
