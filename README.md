All the material is contained in the folder content

GMM.h and GMM.cpp contain a class for managing generic n-dimensional Gaussian
Mixture Model.

The Eigen library is exploited : http://eigen.tuxfamily.org/index.php?title=Main_Page.
Set the folder were you installed Eigen as an additional include directory for compiling.

Training is done using Expectation Maximization, with clusters initialized 
with the k-means classifier.

K means also exists as a utilities of Gaussian_Mixture_Model that can be 
used stand alone.

Two samples are provided: results are produced in c++ and can be visualized launching in Matlab
the appropriate Main.m. 