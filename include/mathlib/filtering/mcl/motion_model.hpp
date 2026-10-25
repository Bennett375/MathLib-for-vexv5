
#pragma once

#include "mathlib/linalg/vector.hpp"
#include "mathlib/linalg/matrix.hpp"
#include "mathlib/linalg/solve_models.hpp"
#include "mathlib/arithmetic/random.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>

namespace math {
namespace filters {
namespace mcl {

class GaussianMotionModel {
public:
    typedef std::function<solver::SolveStatus(
        const linalg::Vector<double>& state,
        const linalg::Vector<double>& control,
        linalg::Vector<double>& predicted_state
    )> TransitionFunction;

    GaussianMotionModel(
        std::size_t state_dimension,
        std::size_t control_dimension,
        const TransitionFunction& transition,
        const linalg::Matrix<double>& covariance
    )
        : state_dimension_(state_dimension),
          control_dimension_(control_dimension),
          transition_(transition),
          covariance_(covariance),
          factor_(linalg::Matrix<double>::zeros(
              state_dimension, state_dimension)),
          status_(solver::SolveStatus::NumericalFailure) {
        status_ = factorize_covariance();
    }

    solver::SolveStatus status() const {
        return status_;
    }

    bool valid() const {
        return status_ == solver::SolveStatus::Success;
    }

    std::size_t state_dim() const {
        return state_dimension_;
    }

    std::size_t control_dim() const {
        return control_dimension_;
    }

    const linalg::Matrix<double>& covariance() const {
        return covariance_;
    }

    solver::SolveStatus operator()(
        const linalg::Vector<double>& state,
        const linalg::Vector<double>& control,
        random::RandomSource& random,
        linalg::Vector<double>& next_state
    ) const {
        using solver::SolveStatus;

        if (!valid()) {
            return status_;
        }

        if (state.size() != state_dimension_ ||
            control.size() != control_dimension_) {
            return SolveStatus::InvalidDimensions;
        }

        if (!finite_vector(state) || !finite_vector(control)) {
            return SolveStatus::NonFiniteInput;
        }

        linalg::Vector<double> predicted =
            linalg::Vector<double>::zeros(state_dimension_);

        const SolveStatus transition_status =
            transition_(state, control, predicted);

        if (transition_status != SolveStatus::Success) {
            return transition_status;
        }

        if (predicted.size() != state_dimension_) {
            return SolveStatus::InvalidDimensions;
        }

        if (!finite_vector(predicted)) {
            return SolveStatus::NonFiniteInput;
        }

        // Draw independent standard-normal samples, then correlate
        // them using the covariance factor: noise = L * z.
        linalg::Vector<double> z =
            linalg::Vector<double>::zeros(state_dimension_);

        for (std::size_t i = 0; i < state_dimension_; ++i) {
            z(i) = random.normal(0.0, 1.0);

            if (!std::isfinite(z(i))) {
                return SolveStatus::NonFiniteInput;
            }
        }

        linalg::Vector<double> candidate =
            linalg::Vector<double>::zeros(state_dimension_);

        for (std::size_t i = 0; i < state_dimension_; ++i) {
            double noise = 0.0;

            for (std::size_t j = 0; j <= i; ++j) {
                noise += factor_(i, j) * z(j);
            }

            candidate(i) = predicted(i) + noise;
        }

        if (!finite_vector(candidate)) {
            return SolveStatus::NumericalFailure;
        }

        // Do not modify the output until the whole operation succeeds.
        next_state = candidate;
        return SolveStatus::Success;
    }

private:
    std::size_t state_dimension_;
    std::size_t control_dimension_;
    TransitionFunction transition_;

    linalg::Matrix<double> covariance_;
    linalg::Matrix<double> factor_;
    solver::SolveStatus status_;

    static bool finite_vector(
        const linalg::Vector<double>& vector
    ) {
        for (std::size_t i = 0; i < vector.size(); ++i) {
            if (!std::isfinite(vector(i))) {
                return false;
            }
        }

        return true;
    }

    solver::SolveStatus factorize_covariance() {
        using solver::SolveStatus;

        if (state_dimension_ == 0 ||
            covariance_.rows() != state_dimension_ ||
            covariance_.cols() != state_dimension_) {
            return SolveStatus::InvalidDimensions;
        }

        if (!transition_) {
            return SolveStatus::UnsupportedMethod;
        }

        double scale = 1.0;

        for (std::size_t i = 0; i < state_dimension_; ++i) {
            for (std::size_t j = 0; j < state_dimension_; ++j) {
                const double value = covariance_(i, j);

                if (!std::isfinite(value)) {
                    return SolveStatus::NonFiniteInput;
                }

                scale = std::max(scale, std::abs(value));
            }
        }

        const double tolerance =
            1e-10 * scale;

        // Require approximate symmetry.
        for (std::size_t i = 0; i < state_dimension_; ++i) {
            for (std::size_t j = i + 1;
                 j < state_dimension_; ++j) {
                if (std::abs(
                        covariance_(i, j) - covariance_(j, i)
                    ) > tolerance) {
                    return SolveStatus::InvalidDimensions;
                }
            }
        }

        factor_ = linalg::Matrix<double>::zeros(
            state_dimension_, state_dimension_);

        // Semidefinite Cholesky factorization.
        // Allows zero-variance dimensions while rejecting
        // materially negative pivots.
        for (std::size_t i = 0; i < state_dimension_; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                double sum = covariance_(i, j);

                for (std::size_t k = 0; k < j; ++k) {
                    sum -= factor_(i, k) * factor_(j, k);
                }

                if (i == j) {
                    if (sum < -tolerance) {
                        return SolveStatus::NotPositiveDefinite;
                    }

                    factor_(i, j) =
                        std::sqrt(std::max(0.0, sum));
                } else {
                    const double pivot = factor_(j, j);

                    if (pivot > std::sqrt(tolerance)) {
                        factor_(i, j) = sum / pivot;
                    } else {
                        if (std::abs(sum) > tolerance) {
                            return SolveStatus::NotPositiveDefinite;
                        }

                        factor_(i, j) = 0.0;
                    }
                }
            }
        }

        return SolveStatus::Success;
    }
};

} // namespace mcl
} // namespace filters
} // namespace math
