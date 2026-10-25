
#pragma once

#include "matrix.hpp"
#include "vector.hpp"
#include "solve_models.hpp"

namespace math {
namespace linalg {

struct GaussianValueResult {
    solver::SolveStatus status;
    double value;

    GaussianValueResult()
        : status(solver::SolveStatus::NumericalFailure),
          value(0.0) {}

    GaussianValueResult(solver::SolveStatus s, double v)
        : status(s), value(v) {}

    bool success() const {
        return status == solver::SolveStatus::Success;
    }
};

class Gaussian {
public:
    Gaussian();

    // Returns Success only if the distribution is valid.
    static solver::SolveStatus create(
        const Vector<double>& mu,
        const Matrix<double>& sigma,
        Gaussian& out);

    bool valid() const;
    std::size_t dim() const;

    const Vector<double>& mean() const;
    const Matrix<double>& covariance() const;

    GaussianValueResult mahalanobis_squared(
        const Vector<double>& x) const;

    GaussianValueResult mahalanobis_distance(
        const Vector<double>& x) const;

    GaussianValueResult log_pdf(
        const Vector<double>& x) const;

    GaussianValueResult pdf(
        const Vector<double>& x) const;

private:
    Vector<double> mu_;
    Matrix<double> sigma_;
    Matrix<double> L_;
    double log_det_sigma_;
    bool valid_;

    solver::SolveStatus factorize_covariance();
};

} // namespace linalg
} // namespace math
