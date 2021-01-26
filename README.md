This libary contains the functionalities required to train and handle **Gaussian Mixture Models**, aka **GMM**.
If you believe to be not really familiar with this object, have a look at ./doc/Gaussian_Mixture_Model.pdf.

**GMM** are trained using the [**Expectation Maximization**](https://stephens999.github.io/fiveMinuteStats/intro_to_em.html) algorithm.
It is possible to specify the initial clusters from which the iterations of the **Expectation Maximization** start. 
Otherwise, when non specifying anything, the [**k-means**](https://en.wikipedia.org/wiki/K-means_clustering) is internally called to create the starting clusters.
**k-means** is also exposed as a callable stand-alone algorithm.

This package is completely **cross-platform**: use [CMake](https://cmake.org) to configure the project containig the libary and some samples.

This library uses [**Eigen**](http://eigen.tuxfamily.org/index.php?title=Main_Page) as linear algebra engine. 
However **Eigen** does not come with this libary: you have to externally download it and set the CMake option variable **EIGEN3_FOLDER** equal to
the folder where you put **Eigen**. You can easily use the Cmake [GUI](https://www.youtube.com/watch?v=LxHV-KNEG3k&t=1s) for doing that.
