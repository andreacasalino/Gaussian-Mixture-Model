/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianUtils/GaussianDistribution.h>

namespace gauss::gmm {
    double evaluate_log_density(const Eigen::VectorXd& point, const std::vector<double>& weights, const std::vector<const GaussianDistribution*>& distributions);
}
