![binaries_compilation](https://github.com/andreacasalino/Gaussian-Mixture-Model/actions/workflows/installArtifacts.yml/badge.svg)
![binaries_compilation](https://github.com/andreacasalino/Gaussian-Mixture-Model/actions/workflows/runTests.yml/badge.svg)

This libary contains the functionalities required to train and handle **Gaussian Mixture Models**, aka **GMM**.
If you believe to be not really familiar with this object, have a look at ./doc/Gaussian_Mixture_Model.pdf.

**GMM** are trained using the [**Expectation Maximization**](https://stephens999.github.io/fiveMinuteStats/intro_to_em.html) algorithm.
It is possible to specify the initial clusters from which the iterations of the **Expectation Maximization** start. 
Otherwise, when non specifying anything, the [**k-means**](https://en.wikipedia.org/wiki/K-means_clustering) is internally called to create the starting clusters.
**k-means** is also exposed as a callable stand-alone algorithm.

This package is completely **cross-platform**: use [CMake](https://cmake.org) to configure the project containig the libary and some samples.

This library uses [**Eigen**](https://gitlab.com/libeigen/eigen) as internal linear algebra engine. 
**Eigen** is by default [fetched](https://cmake.org/cmake/help/latest/module/FetchContent.html) and copied by **CMake** from the latest version on the official **Eigen** repository.
However, you can also use a local version, by [setting](https://www.youtube.com/watch?v=LxHV-KNEG3k&t=1s) the **CMake** option **EIGEN_INSTALL_FOLDER** equal to the root folder storing the local **Eigen** you want to use.

If you have found this library useful, take the time to leave a star ;)
