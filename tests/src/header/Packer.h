#pragma once

#include <Eigen/Core>
#include <vector>

namespace gauss::test {
Eigen::VectorXd make_vector(const std::vector<double> &values);

Eigen::MatrixXd make_matrix(const std::vector<std::vector<double>> &values);
} // namespace gauss::test