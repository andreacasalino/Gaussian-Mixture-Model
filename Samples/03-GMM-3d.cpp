/**
 * Author:    Andrea Casalino
 * Created:   03.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "Utils.h"

/////////////////////////
// GMM in 3 dimensions //
/////////////////////////
int main() {
//sample a random 3d GMM model
	size_t N_clusters = 6;
	gmm::GMM random_model(N_clusters, 3);

//get samples from the random model
	list<gmm::V> train_set = random_model.drawSamples(1000);

//fit a model considering the sampled train set
	gmm::GMM learnt_model(N_clusters, gmm::TrainSet(train_set));

//log the two models to visually check the differences
	print(random_model, "random_model3d.json");
	print(learnt_model, "learnt_model3d.json");

//use the python script Visualize03.py to see the results

	return EXIT_SUCCESS;
}
