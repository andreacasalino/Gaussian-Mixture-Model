/**
 * Author:    Andrea Casalino
 * Created:   03.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianMixtureModel/GaussianMixtureModel.h>
#include <fstream>
#include <iostream>
#include <sstream>

/////////////////// export ///////////////////

std::string toJSON(const Eigen::VectorXd &vector) {
  std::stringstream str;
  str << '[';
  str << vector(0);
  for (Eigen::Index k = 1; k < vector.size(); ++k) {
    str << ',' << vector(k);
  }
  str << ']';
  return str.str();
};

std::string toJSON(const gauss::gmm::Cluster &cl) {
  std::stringstream str;
  str << std::endl << '{';
  str << std::endl << "\"w\":" << cl.weight;
  str << std::endl << ",\"Mean\":" << toJSON(cl.distribution->getMean());
  auto covariance = cl.distribution->getCovariance();
  str << std::endl << ",\"Covariance\":[";
  str << std::endl << toJSON(covariance.row(0));
  for (Eigen::Index k = 1; k < covariance.rows(); ++k) {
    str << std::endl << ',' << toJSON(covariance.row(k));
  }
  str << std::endl << ']';
  str << std::endl << '}';
  return str.str();
};

std::string toJSON(const gauss::gmm::GaussianMixtureModel &model) {
  const auto &clusters = model.getClusters();
  auto it = clusters.begin();
  std::stringstream str;
  str << '[';
  str << std::endl << toJSON(*it);
  ++it;
  for (it; it != clusters.end(); ++it) {
    str << std::endl << ',' << toJSON(*it);
  }
  str << std::endl << ']';
  return str.str();
}

void print(const gauss::gmm::GaussianMixtureModel &model,
           const std::string &file) {
  std::ofstream f(file);
  f << toJSON(model);
  f.close();
}

/////////////////// import ///////////////////

struct StartEndPos {
  std::size_t start;
  std::size_t end;
};
std::unique_ptr<StartEndPos> find_next(const std::string &json_string,
                                       const std::string &to_find,
                                       const std::size_t start_pos,
                                       const std::size_t end_pos) {
  std::size_t match_counter = 0;
  std::size_t last_start = 0;
  for (std::size_t pos = start_pos; pos < end_pos; ++pos) {
    if (json_string[pos] == to_find[match_counter]) {
      ++match_counter;
      if (match_counter == to_find.size()) {
        return std::make_unique<StartEndPos>(StartEndPos{last_start, pos + 1});
      }
    } else {
      match_counter = 0;
      last_start = pos + 1;
    }
  }
  return nullptr;
};

std::vector<StartEndPos> find_all(const std::string &json_string,
                                  const std::string &to_find,
                                  const std::size_t start_pos) {
  std::vector<StartEndPos> result;
  std::size_t pos = start_pos;
  while (true) {
    auto next_match = find_next(json_string, to_find, pos, json_string.size());
    if (nullptr == next_match) {
      break;
    }
    result.push_back(*next_match);
    pos = next_match->end;
  }
  return result;
};

Eigen::VectorXd fromJSON_vec(const std::string &json_string) {
  auto sep = find_all(json_string, ",", 0);
  sep.push_back(StartEndPos{json_string.size(), json_string.size()});
  Eigen::VectorXd result(sep.size());
  std::size_t start_pos = 0;
  for (std::size_t k = 0; k < sep.size(); ++k) {
    result(k) = atof(std::string(json_string, start_pos, sep[k].start).c_str());
    start_pos = sep[k].end;
  }
  return result;
}

Eigen::MatrixXd fromJSON_covariance(const std::string &json_string) {
  std::string temp = json_string;
  temp.push_back(',');
  temp.push_back('[');
  auto sep = find_all(json_string, "],[", 0);
  Eigen::MatrixXd result(sep.size(), sep.size());
  std::size_t start_pos = 0;
  for (std::size_t k = 0; k < sep.size(); ++k) {
    auto row = fromJSON_vec(std::string(json_string, start_pos, sep[k].start));
    if (row.size() != sep.size()) {
      throw std::runtime_error("Invalid json");
    }
    result.row(k) = row;
    start_pos = sep[k].end;
  }
  return result;
}

std::vector<gauss::gmm::Cluster> fromJSON(const std::string &json_string) {
  auto clusters_pos = find_all(json_string, "\"w\":", 0);
  std::vector<gauss::gmm::Cluster> result;
  std::size_t last_pos;
  for (std::size_t b = 0; b < clusters_pos.size(); ++b) {
    if (b == (clusters_pos.size() - 1)) {
      last_pos = json_string.size();
    } else {
      last_pos = clusters_pos[b + 1].start;
    }

    result.emplace_back();
    // weight
    std::size_t cursor = clusters_pos[b].end;
    auto next_match = find_next(json_string, ",\"Mean\":[", cursor, last_pos);
    if (nullptr == next_match) {
      throw std::runtime_error("Invalid json");
    }
    result.back().weight = std::atof(
        std::string(json_string, cursor, next_match->start - cursor).c_str());
    cursor = next_match->start;
    // mean
    next_match =
        find_next(json_string, "],\"Covariance\":[[", cursor, last_pos);
    if (nullptr == next_match) {
      throw std::runtime_error("Invalid json");
    }
    std::string mean_string =
        std::string(json_string, cursor, next_match->start - cursor).c_str();
    cursor = next_match->start;
    // covariance
    next_match = find_next(json_string, "]}", cursor, last_pos);
    if (nullptr == next_match) {
      throw std::runtime_error("Invalid json");
    }
    std::string cov_string =
        std::string(json_string, cursor, next_match->start - cursor).c_str();
    // build distribution
    result.back().distribution = std::make_unique<gauss::GaussianDistribution>(
        fromJSON_vec(mean_string), fromJSON_covariance(cov_string));
  }
  return result;
};

std::vector<gauss::gmm::Cluster> importJSON(const std::string &file) {
  std::string json;
  {
    std::ifstream importer(file);
    std::stringstream buffer;
    buffer << importer.rdbuf();
    json = buffer.str();
  }
  {
    std::stringstream buffer;
    for (const auto &c : json) {
      if ((c == ' ') || (c == '\n') || (c == '\t')) {
        continue;
      }
      buffer << c;
    }
    json = buffer.str();
  }

  return fromJSON(json);
}