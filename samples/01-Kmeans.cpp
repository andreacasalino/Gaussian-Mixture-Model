/**
 * Author:    Andrea Casalino
 * Created:   03.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "Utils.h"
#include <GaussianMixtureModel/KMeans.h>

void addSamplesFromCircle(std::list<Eigen::VectorXd> &samples, const double &x,
                          const double &y, const double &ray,
                          const std::size_t &N_sample);

////////////////////////////////////////
// show how to use K-means clustering //
////////////////////////////////////////
int main() {
  // sample some points in different region of the spaces
  std::list<Eigen::VectorXd> Samples;
  std::size_t sectors = 5;
  double angle = 0.0;
  for (std::size_t k = 0; k < sectors; ++k) {
    addSamplesFromCircle(Samples, 6.0 * cos(angle), 6.0 * sin(angle), 2.0, 50);
    angle += 2.0 * static_cast<double>(3.14159) / static_cast<double>(sectors);
  }
  gauss::TrainSet set(Samples);

  // perform K_means clustering
  std::vector<std::list<const Eigen::VectorXd *>> clusters;
  gauss::gmm::kMeansClustering(clusters, set, sectors);

  // save data for posterior plotting
  std::ofstream f("K_means_clustering");
  auto it_s = clusters.front().begin();
  Eigen::VectorXd temp;
  std::size_t id = 0;
  for (auto it_cl = clusters.begin(); it_cl != clusters.end(); ++it_cl) {
    for (it_s = it_cl->begin(); it_s != it_cl->end(); ++it_s)
      f << id << " " << (*it_s)->transpose() << std::endl;
    ++id;
  }
  f.close();

  std::cout << "Use the python script Visualize01.py to see the results"
            << std::endl;

  return EXIT_SUCCESS;
}

void addSamplesFromCircle(std::list<Eigen::VectorXd> &samples, const double &x,
                          const double &y, const double &ray,
                          const std::size_t &N_sample) {
  double r;
  double teta;
  for (std::size_t k = 0; k < N_sample; ++k) {
    r = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * ray;
    teta = 2.0 * static_cast<double>(3.14159) * static_cast<double>(rand()) /
           static_cast<double>(RAND_MAX);
    samples.emplace_back(2);
    samples.back()(0) = x + r * cos(teta);
    samples.back()(1) = y + r * sin(teta);
  }
}
