/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <TrainSet.h>
#include <fstream>

namespace gauss::gmm {
    TrainSet::TrainSet(const Eigen::VectorXd& initialSample) {
        if (0 == initialSample.size()) throw Error("initial sample can't be empty");
        samples.push_back(initialSample);
    }

    std::list<std::string> sliceFragments(const std::string& toSplit) {
        std::istringstream iss(toSplit);
        std::list<std::string> slices;
        while (!iss.eof()) {
            slices.emplace_back(std::string());
            iss >> slices.back();
            if (slices.back().empty()) slices.pop_back();
        }
        return slices;
    }
    std::list<Eigen::VectorXd> importSamples(const std::string& file_to_read) {
        std::list<Eigen::VectorXd> samples;
        std::ifstream f(file_to_read);
        if (!f.is_open()) {
            throw Error("Invalid file to import train set");
        }
        std::string line;
        std::list<std::string> slices;
        std::size_t pos;
        while (!f.eof()) {
            std::getline(f, line);
            // split and convert to numbers
            slices = sliceFragments(line);
            samples.emplace_back(slices.size());
            pos = 0;
            while (!slices.empty()) {
                samples.back()(pos) = std::atof(slices.front().c_str());
                ++pos;
                slices.pop_front();
            }
        }
        return samples;
    }
    TrainSet::TrainSet(const std::string& file_to_read) : TrainSet(importSamples(file_to_read)) {
    }
}