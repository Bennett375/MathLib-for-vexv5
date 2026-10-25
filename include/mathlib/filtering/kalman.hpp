
#pragma once

#include "mathlib/linalg/gaussian.hpp"
#include "mathlib/linalg/solve_models.hpp"

namespace math {
namespace filters {

class KalmanFilter {
public:
    KalmanFilter();

    // Initialize directly from an existing Gaussian distribution.
    solver::SolveStatus initialize(
        const linalg::Gaussian& initial_distribution);

    // Convenience overload: construct the Gaussian internally.
    solver::SolveStatus initialize(
        const linalg::Vector<double>& initial_state,
        const linalg::Matrix<double>& initial_covariance);

    // Predict:
    // x' = F*x
    // P' = F*P*F^T + Q
    solver::SolveStatus predict(
        const linalg::Matrix<double>& F,
        const linalg::Matrix<double>& Q);

    // Predict with a known control input:
    // x' = F*x + B*u
    // P' = F*P*F^T + Q
    solver::SolveStatus predict(
        const linalg::Matrix<double>& F,
        const linalg::Matrix<double>& B,
        const linalg::Vector<double>& u,
        const linalg::Matrix<double>& Q);

    // Correct the estimate using:
    // z = H*x + measurement noise
    solver::SolveStatus update(
        const linalg::Vector<double>& measurement,
        const linalg::Matrix<double>& H,
        const linalg::Matrix<double>& R);

    // Access the full Gaussian state distribution.
    const linalg::Gaussian& distribution() const;

    // Convenience accessors backed by the Gaussian.
    const linalg::Vector<double>& state() const;
    const linalg::Matrix<double>& covariance() const;

    bool initialized() const;
    std::size_t state_dim() const;

    void reset();

private:
    linalg::Gaussian distribution_;
    bool initialized_;
};

} // namespace filters
} // namespace math
