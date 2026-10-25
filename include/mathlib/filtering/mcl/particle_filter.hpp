#pragma once

#include "mathlib/linalg/vector.hpp"
#include "mathlib/linalg/matrix.hpp"
#include "mathlib/linalg/solve_models.hpp"
#include "mathlib/filtering/mcl/particle.hpp"
#include "mathlib/arithmetic/random.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace math {
namespace filters {
namespace mcl {

struct ParticleFilterOptions {
    std::size_t particle_count;
    double resample_ess_ratio;
    double min_weight;
    bool resample_after_update;

    ParticleFilterOptions()
        : particle_count(500),
          resample_ess_ratio(0.5),
          min_weight(0.0),
          resample_after_update(true) {}
};

struct ParticleEstimate {
    solver::SolveStatus status;
    linalg::Vector<double> mean;
    linalg::Matrix<double> covariance;
    double effective_sample_size;

    explicit ParticleEstimate(std::size_t n)
        : status(solver::SolveStatus::NumericalFailure),
          mean(linalg::Vector<double>::zeros(n)),
          covariance(linalg::Matrix<double>::zeros(n, n)),
          effective_sample_size(0.0) {}

    bool success() const {
        return status == solver::SolveStatus::Success;
    }
};

class ParticleFilter {
public:
    // Produces a new particle state from the old state and control.
    typedef std::function<solver::SolveStatus(
        const linalg::Vector<double>& state,
        const linalg::Vector<double>& control,
        random::RandomSource& random,
        linalg::Vector<double>& next_state
    )> MotionModel;

    // Returns log p(measurement | particle state).
    // The callback writes the result to log_likelihood.
    typedef std::function<solver::SolveStatus(
        const linalg::Vector<double>& state,
        const linalg::Vector<double>& measurement,
        double& log_likelihood
    )> LogLikelihoodModel;

    explicit ParticleFilter(
        random::RandomSource& random,
        const ParticleFilterOptions& options =
            ParticleFilterOptions());

    solver::SolveStatus initialize(
        const std::vector<Particle>& particles);

    // Generate an initial population from a caller-defined sampler.
    typedef std::function<solver::SolveStatus(
        random::RandomSource& random,
        linalg::Vector<double>& state
    )> InitialStateSampler;

    solver::SolveStatus initialize_random(
        std::size_t state_dimension,
        const InitialStateSampler& sampler);

    solver::SolveStatus predict(
        const linalg::Vector<double>& control,
        const MotionModel& motion_model);

    solver::SolveStatus update(
        const linalg::Vector<double>& measurement,
        const LogLikelihoodModel& likelihood_model);

    solver::SolveStatus normalize_weights();

    solver::SolveStatus resample();

    ParticleEstimate estimate() const;

    double effective_sample_size() const;

    std::size_t particle_count() const;
    std::size_t state_dim() const;

    const std::vector<Particle>& particles() const;

    void reset();

private:
    random::RandomSource& random_;
    ParticleFilterOptions options_;
    std::vector<Particle> particles_;
    std::size_t state_dimension_;
    bool initialized_;

    solver::SolveStatus validate_particles(
        const std::vector<Particle>& particles) const;
};

}
}
}