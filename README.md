All the material is contained in the folder content

GMM/GMM.h and GMM/GMM.cpp contain a class for managing generic n-dimensional Gaussian
Mixture Model (GMM).
Gaussian_Mixture_Model.pdf resumes the main generalities about GMM, useful for understanding
the functionalities proposed.

The Eigen library is exploited : http://eigen.tuxfamily.org/index.php?title=Main_Page.
Set the folder were you installed Eigen as an additional include directory when compiling.

Training is done using Expectation Maximization, with clusters initialized 
with the k-means classifier.
K means also exists as a utilities of Gaussian_Mixture_Model that can be 
used stand alone.

Samples are provided in:
Sample_01/Sample_01.cpp
Sample_02/Sample_02.cpp
Sample_03/Sample_03.cpp
Sample_04/Sample_04.cpp

In the folders of the Samples you can find also some python scripts for visualize the results