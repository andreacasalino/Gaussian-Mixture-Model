/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <header/TrainSet.h>
#include <fstream>

namespace gmm {
    TrainSet::TrainSet(const V& initialSample) {
        if (0 == initialSample.size()) throw Error("initial sample can't be empty");
        this->Samples.push_back(initialSample);
    }

    TrainSet& TrainSet::operator==(const TrainSet& o) {
        if (o.Samples.front().size() != this->Samples.front().size()) {
            throw Error("Train sets have different samples size");
        }
        this->addSamples(o.Samples.begin(), o.Samples.end());
        return *this;
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
    std::list<V> importSamples(const std::string& file_to_read) {
        std::list<V> samples;
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