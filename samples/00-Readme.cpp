#include <GaussianMixtureModel/GaussianMixtureModel.h>
#include <GaussianMixtureModel/ExpectationMaximization.h>
#include <GaussianMixtureModel/GaussianMixtureModelFactory.h>

int main() {
	{
		// A gaussian mixture model (gmm) is made of clusters,
		// which are basically gaussian distribution with an associated weight.
		//
		// You can create a gmm by firstly defining the clusters.
		std::vector<gauss::gmm::Cluster> clusters;
		// add the first cluster
		Eigen::VectorXd cluster_mean = ; // fill the mean values
		Eigen::MatrixXd cluster_covariance = ; // fill the covariance values
		double cluster_weight = 0.1;
		std::unique_ptr<gauss::GaussianDistribution> cluster_distributon = std::make_unique<gauss::GaussianDistribution>(cluster_mean, cluster_covariance);
		clusters.push_back(gauss::gmm::Cluster{ cluster_weight, std::move(cluster_distributon) });
		// similarly, add the second and all the others cluster
		clusters.push_back(...);
		// now that the clusters are defined, build the gmm
		gauss::gmm::GaussianMixtureModel gmm_model(clusters);
	}

	{
		// the samples from which the gmm should be deduced
		std::vector<Eigen::VectorXd> samples;
		// apply expectation maximization (EM) to compute the set of clusters that
		// best fit the given samples.
		// The number of expected clusters should be specified
		const std::size_t clusters_size = 4;
		std::vector<gauss::gmm::Cluster> clusters = gauss::gmm::ExpectationMaximization(samples, clusters_size);
		// use the computed clusters to build a gmm
		gauss::gmm::GaussianMixtureModel gmm_model(clusters);
	}

	{
		gauss::gmm::GaussianMixtureModel gmm_model;
		std::vector<Eigen::VectorXd> samples = gmm_model.drawSamples(5000);
	}

	{
		const std::size_t space_size = 4;
		const std::size_t clusters_size = 3;
		gauss::gmm::GaussianMixtureModelFactory model_factory(space_size, clusters_size); // this factory will generate model living in R^4, adopting 3 random clusters
		std::unique_ptr<gauss::gmm::GaussianMixtureModel> random_gmm_model = model_factory.makeRandomModel();
	}

	return EXIT_SUCCESS;
}
