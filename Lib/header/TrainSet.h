/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#ifndef GMM_TRAINSET_H
#define GMM_TRAINSET_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include "Error.h"

namespace gmm {
	typedef Eigen::VectorXd V;

	/** \brief This object can be used to train GMM models
	*/
	class TrainSet {
	public:
		TrainSet(const TrainSet& o) = default;
		TrainSet& operator==(const TrainSet& o);

		TrainSet(TrainSet&& o) = delete;
		TrainSet& operator==(TrainSet&& o) = delete;

		/** \brief Import the training set from a textual file.
		* \details Every line of the file is a sample. Clearly, all the rows must have the same number of elements
		* @param[in] file_to_read The location of the file to import (it can be a relative or an absolute path)
		*/
		TrainSet(const std::string& file_to_read);

		/** \brief Build a training set by cloning the samples passed as input
		* \details Every element of the list is a sample. Clearly, all the element must have the same size
		* @param[in] Samples The list of samples characterizing the training set to build
		*/
		template<typename Collection>
		TrainSet(const Collection& samples) 
			: TrainSet(samples.front()) {
			if (samples.empty()) throw Error("training set can't be empty");
			auto it = samples.begin();
			++it;
			this->addSamples(it, samples.end());
		};

		/** \brief Append the samples of another set into this set.
		\details The initial guess (see Train_set::Set_initial_guess(const std::list<size_t>& cluster_initial_guess) ) is invalidated.
		* @param[in] to_append The training set whose samples must be appended to this set
		*/
		inline void operator+=(const TrainSet& o) { this->addSamples(o.Samples.begin(), o.Samples.end()); };

		/** \brief Get the list of samples of this set.
		*/
		inline const std::list<V>& GetSamples() const { return this->Samples; };

	private:
		TrainSet(const V& initialSample);

		// Collection is an iterable container of V elements
		template<typename CollectionIter>
		void addSamples(const CollectionIter& samplesBegin, const CollectionIter& samplesEnd) {
			std::size_t pos = 0;
			for (auto it = samplesBegin; it != samplesEnd; ++it) {
				if (this->Samples.begin()->size() != it->size()) {
					throw Error("found invalid size for sample " + pos);
				}
				this->Samples.push_back(*it);
				++pos;
			}
		};

		std::list<V>								 Samples;
	};
}

#endif
