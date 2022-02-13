![binaries_compilation](https://github.com/andreacasalino/Gaussian-Mixture-Model/actions/workflows/installArtifacts.yml/badge.svg)
![binaries_compilation](https://github.com/andreacasalino/Gaussian-Mixture-Model/actions/workflows/runTests.yml/badge.svg)

This libary contains the functionalities required to train and handle **Gaussian Mixture Models**, aka **GMM**.
If you believe to be not really familiar with this object, have a look at ./doc/Gaussian_Mixture_Model.pdf.

The construction of a **GMM** can be done by explicitly define the clusters:
```cpp
#include <GaussianMixtureModel/GaussianMixtureModel.h>

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
```

Or by traininig using the [**Expectation Maximization**](https://stephens999.github.io/fiveMinuteStats/intro_to_em.html) algorithm:
```cpp
// the samples from which the gmm should be deduced
std::vector<Eigen::VectorXd> samples;
// apply expectation maximization (EM) to compute the set of clusters that
// best fit the given samples.
// The number of expected clusters should be specified
const std::size_t clusters_size = 4;
std::vector<gauss::gmm::Cluster> clusters = gauss::gmm::ExpectationMaximization(samples, clusters_size);
// use the computed clusters to build a gmm
gauss::gmm::GaussianMixtureModel gmm_model(clusters);
```
It is also possible to specify the initial clusters from which the iterations of the **Expectation Maximization** start. 
Otherwise, when non specifying anything, the [**k-means**](https://en.wikipedia.org/wiki/K-means_clustering) is internally called to create the starting clusters.
**k-means** is also exposed as a callable stand-alone algorithm.

You can also draw samples from an already built **GMM**:
```cpp
std::vector<Eigen::VectorXd> samples = gmm_model.drawSamples(5000)
```

And generate a completely random **GMM**:
```cpp
#include <GaussianMixtureModel/GaussianMixtureModelFactory.h>

const std::size_t space_size = 4;
const std::size_t clusters_size = 3;
gauss::gmm::GaussianMixtureModelFactory model_factory(space_size, clusters_size); // this factory will generate model living in R^4, adopting 3 random clusters
std::unique_ptr<gauss::gmm::GaussianMixtureModel> random_gmm_model = model_factory.makeRandomModel();
```

This package is completely **cross-platform**: use [CMake](https://cmake.org) to configure the project containig the libary and some samples.

This library uses [**Eigen**](https://gitlab.com/libeigen/eigen) as internal linear algebra engine. 
**Eigen** is by default [fetched](https://cmake.org/cmake/help/latest/module/FetchContent.html) and copied by **CMake** from the latest version on the official **Eigen** repository.
However, you can also use a local version, by [setting](https://www.youtube.com/watch?v=LxHV-KNEG3k&t=1s) the **CMake** option **EIGEN_INSTALL_FOLDER** equal to the root folder storing the local **Eigen** you want to use.

If you have found this library useful, take the time to leave a star ;)
