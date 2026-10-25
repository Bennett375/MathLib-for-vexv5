
#include "gaussian.hpp"

#include <cmath>
#include <limits>

namespace math {
namespace linalg {

namespace {

const double LOG_TWO_PI =
    1.83787706640934548356;

} // namespace

Gaussian::Gaussian()
    : mu_(0),
      sigma_(0, 0),
      L_(0, 0),
      log_det_sigma_(0.0),
      valid_(false) {}

solver::SolveStatus Gaussian::create(
    const Vector<double>& mu,
    const Matrix<double>& sigma,
    Gaussian& out) {

    const std::size_t n = mu.size();

    if (n == 0 ||
        sigma.rows() != n ||
        sigma.cols() != n) {
        out.valid_ = false;
        return solver::SolveStatus::InvalidDimensions;
    }

    out.mu_ = mu;
    out.sigma_ = sigma;
    out.L_ = Matrix<double>(n, n, 0.0);
    out.log_det_sigma_ = 0.0;
    out.valid_ = false;

    const solver::SolveStatus status =
        out.factorize_covariance();

    if (status != solver::SolveStatus::Success) {
        return status;
    }

    out.valid_ = true;
    return solver::SolveStatus::Success;
}

solver::SolveStatus Gaussian::factorize_covariance() {
    solver::SolveOptions<double> options;

    const solver::CholeskyResult<double> result =
        solver::cholesky_decompose(sigma_, options);

    if (result.status != solver::SolveStatus::Success) {
        return result.status;
    }

    L_ = result.L;

    double log_det = 0.0;

    for (std::size_t i = 0; i < dim(); ++i) {
        const double diagonal = L_(i, i);

        if (!(diagonal > 0.0) || !std::isfinite(diagonal)) {
            return solver::SolveStatus::NumericalFailure;
        }

        log_det += 2.0 * std::log(diagonal);
    }

    if (!std::isfinite(log_det)) {
        return solver::SolveStatus::NumericalFailure;
    }

    log_det_sigma_ = log_det;
    return solver::SolveStatus::Success;
}

bool Gaussian::valid() const {
    return valid_;
}

std::size_t Gaussian::dim() const {
    return mu_.size();
}

const Vector<double>& Gaussian::mean() const {
    return mu_;
}

const Matrix<double>& Gaussian::covariance() const {
    return sigma_;
}

GaussianValueResult Gaussian::mahalanobis_squared(
    const Vector<double>& x) const {

    if (!valid_) {
        return {solver::SolveStatus::NumericalFailure, 0.0};
    }

    if (x.size() != dim()) {
        return {solver::SolveStatus::InvalidDimensions, 0.0};
    }

    const std::size_t n = dim();
    Vector<double> y(n, 0.0);

    // Solve L*y = x - mu by forward substitution.
    for (std::size_t i = 0; i < n; ++i) {
        double value = x(i) - mu_(i);

        for (std::size_t j = 0; j < i; ++j) {
            value -= L_(i, j) * y(j);
        }

        const double diagonal = L_(i, i);

        if (!(diagonal > 0.0) || !std::isfinite(diagonal)) {
            return {solver::SolveStatus::NumericalFailure, 0.0};
        }

        y(i) = value / diagonal;
    }

    double squared_distance = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        squared_distance += y(i) * y(i);
    }

    if (!std::isfinite(squared_distance)) {
        return {solver::SolveStatus::NumericalFailure, 0.0};
    }

    return {solver::SolveStatus::Success, squared_distance};
}

GaussianValueResult Gaussian::mahalanobis_distance(
    const Vector<double>& x) const {

    const GaussianValueResult result =
        mahalanobis_squared(x);

    if (!result.success()) {
        return result;
    }

    return {
        solver::SolveStatus::Success,
        std::sqrt(result.value)
    };
}

GaussianValueResult Gaussian::log_pdf(
    const Vector<double>& x) const {

    const GaussianValueResult distance =
        mahalanobis_squared(x);

    if (!distance.success()) {
        return distance;
    }

    const double n = static_cast<double>(dim());

    const double value = -0.5 * (
        n * LOG_TWO_PI +
        log_det_sigma_ +
        distance.value
    );

    if (!std::isfinite(value)) {
        return {solver::SolveStatus::NumericalFailure, 0.0};
    }

    return {solver::SolveStatus::Success, value};
}

GaussianValueResult Gaussian::pdf(
    const Vector<double>& x) const {

    const GaussianValueResult result = log_pdf(x);

    if (!result.success()) {
        return result;
    }

    // Underflow to zero is valid for a very small density.
    const double density = std::exp(result.value);

    return {solver::SolveStatus::Success, density};
}

} // namespace linalg
} // namespace math
