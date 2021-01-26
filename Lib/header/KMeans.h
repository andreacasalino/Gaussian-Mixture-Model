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
	/** @brief K means clustering.
	 * Clusters are initialized with the Forgy method (https://en.wikipedia.org/wiki/K-means_clustering).
	 * @param[out] results of the K means clustering. Every element in the list is a cluster, represented by the list of samples pertaining to that cluster (const pointers points to elements in Samples).
	 * @param[in] the samples to clusterize
	 * @param[in] the number of clusters to assume
	 * @param[in] the maximum number of iterations to assume (the algorithm can reach a convergence also before)
	 */
	void kMeansClustering(std::vector<std::list<const V*>>& clusters, const TrainSet& Samples, const std::size_t& N_cluster, const std::size_t& Iterations = 1000);
}

#endif