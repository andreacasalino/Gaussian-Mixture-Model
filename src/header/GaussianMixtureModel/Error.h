/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <stdexcept>

namespace gauss::gmm {
    /** @brief A runtime error that can be raised by objects in this namespace
     */
    class Error : public std::runtime_error {
    public:
        explicit Error(const std::string& what) : std::runtime_error(what) {
        };
    };
}
