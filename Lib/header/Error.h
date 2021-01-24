/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#ifndef GMM_ERROR_H
#define GMM_ERROR_H

#include <stdexcept>

namespace gmm {
    /** @brief A runtime error that can be raised when using any object in gmm::
     */
    class Error : public std::runtime_error {
    public:
        explicit Error(const std::string& what) : std::runtime_error(what) {
        };
    };
}

#endif
