
#include "mathlib/filtering/kalman.hpp"
#include "mathlib/linalg/solve_models.hpp"

#include <cmath>
#include <cstddef>

namespace math {
namespace filters {
namespace {

// Validation

bool symmetric_matrix(const linalg::Matrix<double>& A) {
    if (A.rows() != A.cols()) {
        return false;
    }

    const std::size_t n = A.rows();
    const double tolerance = 1e-9;

    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = r + 1; c < n; ++c) {
            const double a = A(r, c);
            const double b = A(c, r);

            if (std::fabs(a - b) >
                tolerance * (1.0 + std::fabs(a) + std::fabs(b))) {
                return false;
            }
        }
    }

    return true;
}

// Removes small asymmetries introduced by floating-point arithmetic.
linalg::Matrix<double> symmetrize(const linalg::Matrix<double>& A) {
    linalg::Matrix<double> result(A.rows(), A.cols(), 0.0);

    for (std::size_t r = 0; r < A.rows(); ++r) {
        for (std::size_t c = 0; c < A.cols(); ++c) {
            result(r, c) = 0.5 * (A(r, c) + A(c, r));
        }
    }

    return result;
}

// Cholesky factorization and solves
//
// These helpers solve S*x = b without explicitly forming inverse(S).
// S must be symmetric positive definite.

solver::SolveStatus cholesky_decompose(
    const linalg::Matrix<double>& A,
    linalg::Matrix<double>& L) {

    if (A.rows() == 0 || A.rows() != A.cols()) {
        return solver::SolveStatus::InvalidDimensions;
    }

    if (!solver::detail::matrix_is_finite(A)) {
        return solver::SolveStatus::NonFiniteInput;
    }

    if (!symmetric_matrix(A)) {
        return solver::SolveStatus::NumericalFailure;
    }

    const std::size_t n = A.rows();
    L = linalg::Matrix<double>(n, n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = A(i, j);

            for (std::size_t k = 0; k < j; ++k) {
                sum -= L(i, k) * L(j, k);
            }

            if (i == j) {
                if (!std::isfinite(sum)) {
                    return solver::SolveStatus::NumericalFailure;
                }

                if (sum <= 0.0) {
                    return solver::SolveStatus::NotPositiveDefinite;
                }

                L(i, j) = std::sqrt(sum);

                if (!std::isfinite(L(i, j)) || L(i, j) == 0.0) {
                    return solver::SolveStatus::NumericalFailure;
                }
            } else {
                if (L(j, j) == 0.0) {
                    return solver::SolveStatus::NumericalFailure;
                }

                L(i, j) = sum / L(j, j);

                if (!std::isfinite(L(i, j))) {
                    return solver::SolveStatus::NumericalFailure;
                }
            }
        }
    }

    return solver::SolveStatus::Success;
}

solver::SolveStatus solve_cholesky(
    const linalg::Matrix<double>& L,
    const linalg::Vector<double>& b,
    linalg::Vector<double>& x) {

    if (L.rows() == 0 ||
        L.rows() != L.cols() ||
        L.rows() != b.size()) {
        return solver::SolveStatus::InvalidDimensions;
    }

    const std::size_t n = L.rows();
    linalg::Vector<double> y(n, 0.0);
    x = linalg::Vector<double>(n, 0.0);

    // Forward substitution: L*y = b.
    for (std::size_t i = 0; i < n; ++i) {
        double sum = b(i);

        for (std::size_t j = 0; j < i; ++j) {
            sum -= L(i, j) * y(j);
        }

        if (L(i, i) == 0.0 || !std::isfinite(L(i, i))) {
            return solver::SolveStatus::NumericalFailure;
        }

        y(i) = sum / L(i, i);

        if (!std::isfinite(y(i))) {
            return solver::SolveStatus::NumericalFailure;
        }
    }

    // Back substitution: L^T*x = y.
    for (std::size_t ii = n; ii-- > 0;) {
        double sum = y(ii);

        for (std::size_t j = ii + 1; j < n; ++j) {
            sum -= L(j, ii) * x(j);
        }

        if (L(ii, ii) == 0.0 || !std::isfinite(L(ii, ii))) {
            return solver::SolveStatus::NumericalFailure;
        }

        x(ii) = sum / L(ii, ii);

        if (!std::isfinite(x(ii))) {
            return solver::SolveStatus::NumericalFailure;
        }
    }

    return solver::SolveStatus::Success;
}

// Computes K = P*H^T*S^-1 by solving S*k_i^T = (P*H^T)_i^T
// for each row i of K. No matrix inverse is formed.
solver::SolveStatus compute_kalman_gain(
    const linalg::Matrix<double>& P,
    const linalg::Matrix<double>& H,
    const linalg::Matrix<double>& S,
    linalg::Matrix<double>& K) {

    const std::size_t n = P.rows();
    const std::size_t m = H.rows();

    linalg::Matrix<double> PHt = P * H.transpose();
    linalg::Matrix<double> L;

    solver::SolveStatus status = cholesky_decompose(S, L);

    if (status != solver::SolveStatus::Success) {
        return status;
    }

    K = linalg::Matrix<double>(n, m, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        linalg::Vector<double> rhs(m, 0.0);

        for (std::size_t j = 0; j < m; ++j) {
            rhs(j) = PHt(i, j);
        }

        linalg::Vector<double> solution(m, 0.0);

        status = solve_cholesky(L, rhs, solution);

        if (status != solver::SolveStatus::Success) {
            return status;
        }

        for (std::size_t j = 0; j < m; ++j) {
            K(i, j) = solution(j);
        }
    }

    if (!solver::detail::matrix_is_finite(K)) {
        return solver::SolveStatus::NumericalFailure;
    }

    return solver::SolveStatus::Success;
}

bool valid_covariance_input(const linalg::Matrix<double>& A) {
    return A.rows() > 0 &&
           A.rows() == A.cols() &&
           solver::detail::matrix_is_finite(A) &&
           symmetric_matrix(A);
}

} // anonymous namespace

// KalmanFilter

KalmanFilter::KalmanFilter()
    : distribution_(),
      initialized_(false) {
}

solver::SolveStatus KalmanFilter::initialize(
    const linalg::Gaussian& initial_distribution) {

    if (!initial_distribution.valid()) {
        return solver::SolveStatus::InvalidDimensions;
    }

    // Gaussian::create is the validation boundary for distributions;
    // reconstruct to ensure this filter accepts only a valid Gaussian.
    linalg::Gaussian candidate;

    solver::SolveStatus status = linalg::Gaussian::create(
        initial_distribution.mean(),
        initial_distribution.covariance(),
        candidate);

    if (status != solver::SolveStatus::Success) {
        return status;
    }

    distribution_ = candidate;
    initialized_ = true;

    return solver::SolveStatus::Success;
}

solver::SolveStatus KalmanFilter::initialize(
    const linalg::Vector<double>& initial_state,
    const linalg::Matrix<double>& initial_covariance) {

    linalg::Gaussian candidate;

    solver::SolveStatus status = linalg::Gaussian::create(
        initial_state,
        initial_covariance,
        candidate);

    if (status != solver::SolveStatus::Success) {
        return status;
    }

    distribution_ = candidate;
    initialized_ = true;

    return solver::SolveStatus::Success;
}

solver::SolveStatus KalmanFilter::predict(
    const linalg::Matrix<double>& F,
    const linalg::Matrix<double>& Q) {

    if (!initialized_) {
        return solver::SolveStatus::InvalidDimensions;
    }

    const std::size_t n = distribution_.dim();

    if (F.rows() != n || F.cols() != n ||
        Q.rows() != n || Q.cols() != n) {
        return solver::SolveStatus::InvalidDimensions;
    }

    if (!solver::detail::matrix_is_finite(F) || !solver::detail::matrix_is_finite(Q)) {
        return solver::SolveStatus::NonFiniteInput;
    }

    if (!valid_covariance_input(Q)) {
        return solver::SolveStatus::NumericalFailure;
    }

    const linalg::Vector<double>& x = distribution_.mean();
    const linalg::Matrix<double>& P = distribution_.covariance();

    // x' = F*x
    linalg::Vector<double> predicted_mean = F * x;

    // P' = F*P*F^T + Q
    linalg::Matrix<double> predicted_covariance =
        (F * P) * F.transpose() + Q;

    predicted_covariance = symmetrize(predicted_covariance);

    if (!solver::detail::vector_is_finite(predicted_mean) ||
        !solver::detail::matrix_is_finite(predicted_covariance)) {
        return solver::SolveStatus::NumericalFailure;
    }

    linalg::Gaussian candidate;

    solver::SolveStatus status = linalg::Gaussian::create(
        predicted_mean,
        predicted_covariance,
        candidate);

    if (status != solver::SolveStatus::Success) {
        return status;
    }

    // Commit only after all calculations and validation succeed.
    distribution_ = candidate;

    return solver::SolveStatus::Success;
}

solver::SolveStatus KalmanFilter::predict(
    const linalg::Matrix<double>& F,
    const linalg::Matrix<double>& B,
    const linalg::Vector<double>& u,
    const linalg::Matrix<double>& Q) {

    if (!initialized_) {
        return solver::SolveStatus::InvalidDimensions;
    }

    const std::size_t n = distribution_.dim();

    if (F.rows() != n || F.cols() != n ||
        B.rows() != n ||
        B.cols() != u.size() ||
        Q.rows() != n || Q.cols() != n) {
        return solver::SolveStatus::InvalidDimensions;
    }

    if (!solver::detail::matrix_is_finite(F) ||
        !solver::detail::matrix_is_finite(B) ||
        !solver::detail::vector_is_finite(u) ||
        !solver::detail::matrix_is_finite(Q)) {
        return solver::SolveStatus::NonFiniteInput;
    }

    if (!valid_covariance_input(Q)) {
        return solver::SolveStatus::NumericalFailure;
    }

    const linalg::Vector<double>& x = distribution_.mean();
    const linalg::Matrix<double>& P = distribution_.covariance();

    // x' = F*x + B*u
    linalg::Vector<double> predicted_mean =
        (F * x) + (B * u);

    // P' = F*P*F^T + Q
    linalg::Matrix<double> predicted_covariance =
        (F * P) * F.transpose() + Q;

    predicted_covariance = symmetrize(predicted_covariance);

    if (!solver::detail::vector_is_finite(predicted_mean) ||
        !solver::detail::matrix_is_finite(predicted_covariance)) {
        return solver::SolveStatus::NumericalFailure;
    }

    linalg::Gaussian candidate;

    solver::SolveStatus status = linalg::Gaussian::create(
        predicted_mean,
        predicted_covariance,
        candidate);

    if (status != solver::SolveStatus::Success) {
        return status;
    }

    distribution_ = candidate;

    return solver::SolveStatus::Success;
}

solver::SolveStatus KalmanFilter::update(
    const linalg::Vector<double>& measurement,
    const linalg::Matrix<double>& H,
    const linalg::Matrix<double>& R) {

    if (!initialized_) {
        return solver::SolveStatus::InvalidDimensions;
    }

    const std::size_t n = distribution_.dim();
    const std::size_t m = measurement.size();

    if (m == 0 ||
        H.rows() != m ||
        H.cols() != n ||
        R.rows() != m ||
        R.cols() != m) {
        return solver::SolveStatus::InvalidDimensions;
    }

    if (!solver::detail::vector_is_finite(measurement) ||
        !solver::detail::matrix_is_finite(H) ||
        !solver::detail::matrix_is_finite(R)) {
        return solver::SolveStatus::NonFiniteInput;
    }

    if (!valid_covariance_input(R)) {
        return solver::SolveStatus::NumericalFailure;
    }

    const linalg::Vector<double>& x = distribution_.mean();
    const linalg::Matrix<double>& P = distribution_.covariance();

    // Innovation: y = z - H*x
    linalg::Vector<double> predicted_measurement = H * x;
    linalg::Vector<double> innovation =
        measurement - predicted_measurement;

    // Innovation covariance: S = H*P*H^T + R
    linalg::Matrix<double> S =
        (H * P) * H.transpose() + R;

    S = symmetrize(S);

    if (!solver::detail::vector_is_finite(innovation) || !solver::detail::matrix_is_finite(S)) {
        return solver::SolveStatus::NumericalFailure;
    }

    // K = P*H^T*S^-1, implemented using Cholesky solves.
    linalg::Matrix<double> K;

    solver::SolveStatus status = compute_kalman_gain(
        P, H, S, K);

    if (status != solver::SolveStatus::Success) {
        return status;
    }

    // Corrected mean: x+ = x + K*y
    linalg::Vector<double> correction = K * innovation;
    linalg::Vector<double> updated_mean = x + correction;

    // Joseph covariance update:
    // P+ = (I-KH)*P*(I-KH)^T + K*R*K^T
    linalg::Matrix<double> I = linalg::Matrix<double>::identity(n);
    linalg::Matrix<double> KH = K * H;
    linalg::Matrix<double> A = I - KH;

    linalg::Matrix<double> first_term = (A * P) * A.transpose();

    linalg::Matrix<double> second_term = (K * R) * K.transpose();

    linalg::Matrix<double> updated_covariance =
        first_term + second_term;

    updated_covariance = symmetrize(updated_covariance);

    if (!solver::detail::vector_is_finite(updated_mean) ||
        !solver::detail::matrix_is_finite(updated_covariance)) {
        return solver::SolveStatus::NumericalFailure;
    }

    // Gaussian::create checks that the resulting distribution is valid.
    linalg::Gaussian candidate;

    status = linalg::Gaussian::create(
        updated_mean,
        updated_covariance,
        candidate);

    if (status != solver::SolveStatus::Success) {
        return status;
    }

    // Do not modify the current distribution until validation succeeds.
    distribution_ = candidate;

    return solver::SolveStatus::Success;
}

const linalg::Gaussian& KalmanFilter::distribution() const {
    return distribution_;
}

const linalg::Vector<double>& KalmanFilter::state() const {
    return distribution_.mean();
}

const linalg::Matrix<double>& KalmanFilter::covariance() const {
    return distribution_.covariance();
}

bool KalmanFilter::initialized() const {
    return initialized_;
}

std::size_t KalmanFilter::state_dim() const {
    return initialized_ ? distribution_.dim() : 0;
}

void KalmanFilter::reset() {
    distribution_ = linalg::Gaussian();
    initialized_ = false;
}

} // namespace filters
} // namespace math
