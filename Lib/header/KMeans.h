/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#ifndef GMM_KMEANS_H
#define GMM_KMEANS_H

#include "TrainSet.h"

namespace gmm {
	/** \brief K means clustering .
	* \details Clusters are initialized with the Forgy method (https://en.wikipedia.org/wiki/K-means_clustering).
	Clusters are initialized with this method when training GMM models with the expectation maximization algorithm.
	* @param[in] Samples the samples to clusterize
	* @param[in] N_cluster the number of clusters to consider
	* @param[in] Iterations the maximum number of iterations to assume (the algorithm can reach a convergence also before)
	* @param[out] clusters results of the K means clustering. Every element in the list is a cluster, represented by the list of samples
	put in that cluster (const pointers points to elements in Samples).
	*/
	void kMeansClustering(std::vector<std::list<const V*>>& clusters, const TrainSet& Samples, const size_t& N_cluster, const size_t& Iterations = 1000);
}

#endif