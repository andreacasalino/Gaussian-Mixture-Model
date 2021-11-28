/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "EvaluateLogDensity.h"

namespace gauss::gmm {
    double evaluate_log_density(const Eigen::VectorXd& point, const std::vector<double>& weights, const std::vector<const GaussianDistribution*>& distributions) {
        double den = 0.0;
        for (std::size_t k = 0; k < weights.size(); ++k) {
            den += exp(log(weights[k]) + distributions[k]->evaluateLogDensity(point));
        }
        den = log(den);
        return den;
    }
}