#include <Packer.h>
#include <stdexcept>

namespace gauss::test {
Eigen::VectorXd make_vector(const std::vector<double> &values) {
  if (values.empty()) {
    throw std::runtime_error("empty buffer");
  }
  Eigen::VectorXd vector(values.size());
  Eigen::Index index = 0;
  for (const auto &value : values) {
    vector(index) = value;
    ++index;
  }
  return vector;
}

Eigen::MatrixXd make_matrix(const std::vector<std::vector<double>> &values) {
  if (values.empty()) {
    throw std::runtime_error("empty buffer");
  }
  std::size_t size = values.size();
  for (std::size_t k = 1; k < size; ++k) {
    if (values[k].size() != size) {
      throw std::runtime_error("invalid buffer");
    }
  }

  Eigen::MatrixXd matrix(size, size);
  for (std::size_t r = 0; r < size; ++r) {
    for (std::size_t c = 0; c < size; ++c) {
      matrix(r, c) = values[r][c];
    }
  }
  return matrix;
}
} // namespace gauss::test