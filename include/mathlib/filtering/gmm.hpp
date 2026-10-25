#pragma once

#include "mathlib/linalg/gaussian.hpp"
#include "mathlib/linalg/solve_models.hpp"

#include <cstddef>
#include <vector>

namespace math {
namespace filters {

struct GaussianComponent {
    linalg::Gaussian distribution;
    double weight;

    GaussianComponent()
        : weight(0.0) {}
};

struct GMMOptions {
    std::size_t max_components;
    double min_weight;
    double merge_threshold;
    double covariance_floor;

    GMMOptions()
        : max_components(8),
          min_weight(0.01),
          merge_threshold(2.0),
          covariance_floor(1e-6) {}
};

struct GMMEstimate {
    solver::SolveStatus status;
    linalg::Vector<double> mean;
    linalg::Matrix<double> covariance;

    GMMEstimate(std::size_t n)
        : status(solver::SolveStatus::NumericalFailure), mean(linalg::Vector<double>::zeros(n)), covariance(linalg::Matrix<double>::zeros(n, n)) {}

    bool success() const {
        return status == solver::SolveStatus::Success;
    }
};

class GaussianMixtureFilter {
public:
    explicit GaussianMixtureFilter(
        const GMMOptions& options = GMMOptions());

    solver::SolveStatus initialize(
        const std::vector<GaussianComponent>& components);

    solver::SolveStatus predict(
        const linalg::Matrix<double>& F,
        const linalg::Matrix<double>& Q);

    solver::SolveStatus update(
        const linalg::Vector<double>& measurement,
        const linalg::Matrix<double>& H,
        const linalg::Matrix<double>& R);

    solver::SolveStatus prune();

    GMMEstimate estimate() const;

    std::size_t component_count() const;
    const std::vector<GaussianComponent>& components() const;
    const GaussianComponent& most_likely_component() const;

private:
    GMMOptions options_;
    std::vector<GaussianComponent> components_;

    void normalize_weights();
    solver::SolveStatus merge_similar_components();
};

} // namespace filters
} // namespace math