
#include "mathlib/filtering/mcl/particle_filter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace math {
namespace filters {
namespace mcl {

namespace {

using linalg::Matrix;
using linalg::Vector;
using solver::SolveStatus;

bool finite_vector(const Vector<double>& v) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (!std::isfinite(v(i))) {
            return false;
        }
    }
    return true;
}

bool valid_options(const ParticleFilterOptions& options) {
    return options.particle_count > 0 &&
           std::isfinite(options.resample_ess_ratio) &&
           options.resample_ess_ratio >= 0.0 &&
           options.resample_ess_ratio <= 1.0 &&
           std::isfinite(options.min_weight) &&
           options.min_weight >= 0.0 &&
           options.min_weight <
               1.0 / static_cast<double>(options.particle_count);
}

// Normalize a candidate population without modifying the live population
// until validation succeeds.
SolveStatus normalize_population(
    std::vector<Particle>& particles,
    double min_weight
) {
    if (particles.empty()) {
        return SolveStatus::InvalidDimensions;
    }

    double total = 0.0;

    for (const Particle& p : particles) {
        if (!std::isfinite(p.weight)) {
            return SolveStatus::NonFiniteInput;
        }

        if (p.weight < 0.0) {
            return SolveStatus::NumericalFailure;
        }

        total += p.weight;
    }

    if (!std::isfinite(total) || total <= 0.0) {
        return SolveStatus::NumericalFailure;
    }

    for (Particle& p : particles) {
        p.weight /= total;
    }

    if (min_weight > 0.0) {
        // The configured floor must be small enough to leave probability
        // mass available for the remaining particles.
        if (min_weight *
                static_cast<double>(particles.size()) >= 1.0) {
            return SolveStatus::NumericalFailure;
        }

        for (Particle& p : particles) {
            p.weight = std::max(p.weight, min_weight);
        }

        total = 0.0;
        for (const Particle& p : particles) {
            total += p.weight;
        }

        if (!std::isfinite(total) || total <= 0.0) {
            return SolveStatus::NumericalFailure;
        }

        for (Particle& p : particles) {
            p.weight /= total;
        }
    }

    return SolveStatus::Success;
}

double compute_ess(const std::vector<Particle>& particles) {
    double sum_squares = 0.0;

    for (const Particle& p : particles) {
        sum_squares += p.weight * p.weight;
    }

    if (!std::isfinite(sum_squares) || sum_squares <= 0.0) {
        return 0.0;
    }

    return 1.0 / sum_squares;
}

} // namespace

ParticleFilter::ParticleFilter(
    random::RandomSource& random,
    const ParticleFilterOptions& options
)
    : random_(random),
      options_(options),
      particles_(),
      state_dimension_(0),
      initialized_(false) {}

solver::SolveStatus ParticleFilter::validate_particles(
    const std::vector<Particle>& particles
) const {
    if (particles.empty()) {
        return solver::SolveStatus::InvalidDimensions;
    }

    const std::size_t dimension = particles.front().state.size();

    if (dimension == 0) {
        return solver::SolveStatus::InvalidDimensions;
    }

    double total_weight = 0.0;

    for (const Particle& p : particles) {
        if (p.state.size() != dimension) {
            return solver::SolveStatus::InvalidDimensions;
        }

        if (!finite_vector(p.state) || !std::isfinite(p.weight)) {
            return solver::SolveStatus::NonFiniteInput;
        }

        if (p.weight < 0.0) {
            return solver::SolveStatus::NumericalFailure;
        }

        total_weight += p.weight;
    }

    if (!std::isfinite(total_weight) || total_weight <= 0.0) {
        return solver::SolveStatus::NumericalFailure;
    }

    return solver::SolveStatus::Success;
}

solver::SolveStatus ParticleFilter::initialize(
    const std::vector<Particle>& particles
) {
    if (!valid_options(options_)) {
        return solver::SolveStatus::NumericalFailure;
    }

    const solver::SolveStatus validation =
        validate_particles(particles);

    if (validation != solver::SolveStatus::Success) {
        return validation;
    }

    if (particles.size() != options_.particle_count) {
        return solver::SolveStatus::InvalidDimensions;
    }

    std::vector<Particle> candidate = particles;

    const solver::SolveStatus normalization =
        normalize_population(candidate, options_.min_weight);

    if (normalization != solver::SolveStatus::Success) {
        return normalization;
    }

    state_dimension_ = candidate.front().state.size();
    particles_.swap(candidate);
    initialized_ = true;

    return solver::SolveStatus::Success;
}

solver::SolveStatus ParticleFilter::initialize_random(
    std::size_t state_dimension,
    const InitialStateSampler& sampler
) {
    if (!valid_options(options_) ||
        state_dimension == 0 ||
        !sampler) {
        return solver::SolveStatus::InvalidDimensions;
    }

    std::vector<Particle> candidate;
    candidate.reserve(options_.particle_count);

    const double uniform_weight =
        1.0 / static_cast<double>(options_.particle_count);

    for (std::size_t i = 0; i < options_.particle_count; ++i) {
        Particle p(state_dimension);
        p.weight = uniform_weight;

        const solver::SolveStatus status =
            sampler(random_, p.state);

        if (status != solver::SolveStatus::Success) {
            return status;
        }

        if (p.state.size() != state_dimension) {
            return solver::SolveStatus::InvalidDimensions;
        }

        if (!finite_vector(p.state)) {
            return solver::SolveStatus::NonFiniteInput;
        }

        candidate.push_back(std::move(p));
    }

    const solver::SolveStatus normalization =
        normalize_population(candidate, options_.min_weight);

    if (normalization != solver::SolveStatus::Success) {
        return normalization;
    }

    state_dimension_ = state_dimension;
    particles_.swap(candidate);
    initialized_ = true;

    return solver::SolveStatus::Success;
}

solver::SolveStatus ParticleFilter::predict(
    const Vector<double>& control,
    const MotionModel& motion_model
) {
    if (!initialized_) {
        return solver::SolveStatus::NumericalFailure;
    }

    if (!motion_model) {
        return solver::SolveStatus::UnsupportedMethod;
    }

    if (!finite_vector(control)) {
        return solver::SolveStatus::NonFiniteInput;
    }

    std::vector<Particle> candidate = particles_;

    for (Particle& p : candidate) {
        Vector<double> next_state =
            Vector<double>::zeros(state_dimension_);

        const solver::SolveStatus status =
            motion_model(p.state, control, random_, next_state);

        if (status != solver::SolveStatus::Success) {
            return status;
        }

        if (next_state.size() != state_dimension_) {
            return solver::SolveStatus::InvalidDimensions;
        }

        if (!finite_vector(next_state)) {
            return solver::SolveStatus::NonFiniteInput;
        }

        p.state = std::move(next_state);
    }

    particles_.swap(candidate);
    return solver::SolveStatus::Success;
}

solver::SolveStatus ParticleFilter::update(
    const Vector<double>& measurement,
    const LogLikelihoodModel& likelihood_model
) {
    if (!initialized_) {
        return solver::SolveStatus::NumericalFailure;
    }

    if (!likelihood_model) {
        return solver::SolveStatus::UnsupportedMethod;
    }

    if (!finite_vector(measurement)) {
        return solver::SolveStatus::NonFiniteInput;
    }

    std::vector<Particle> candidate = particles_;
    std::vector<double> log_weights(candidate.size());

    double max_log_weight =
        -std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < candidate.size(); ++i) {
        double log_likelihood =
            -std::numeric_limits<double>::infinity();

        const solver::SolveStatus status = likelihood_model(
            candidate[i].state,
            measurement,
            log_likelihood
        );

        if (status != solver::SolveStatus::Success) {
            return status;
        }

        // Negative infinity represents a zero likelihood. NaN and
        // positive infinity are rejected.
        if (std::isnan(log_likelihood) ||
            log_likelihood ==
                std::numeric_limits<double>::infinity()) {
            return solver::SolveStatus::NonFiniteInput;
        }

        if (candidate[i].weight <= 0.0 ||
            log_likelihood ==
                -std::numeric_limits<double>::infinity()) {
            log_weights[i] =
                -std::numeric_limits<double>::infinity();
        } else {
            log_weights[i] =
                std::log(candidate[i].weight) + log_likelihood;
        }

        if (log_weights[i] > max_log_weight) {
            max_log_weight = log_weights[i];
        }
    }

    // Every particle assigned zero posterior probability.
    if (!std::isfinite(max_log_weight)) {
        return solver::SolveStatus::NumericalFailure;
    }

    // Log-sum-exp stabilization: exp(log_weight - max_log_weight)
    // avoids underflow from very small raw likelihoods.
    double scaled_total = 0.0;

    for (std::size_t i = 0; i < candidate.size(); ++i) {
        const double w = std::exp(
            log_weights[i] - max_log_weight
        );

        candidate[i].weight = w;
        scaled_total += w;
    }

    if (!std::isfinite(scaled_total) || scaled_total <= 0.0) {
        return solver::SolveStatus::NumericalFailure;
    }

    for (Particle& p : candidate) {
        p.weight /= scaled_total;
    }

    const solver::SolveStatus normalization =
        normalize_population(candidate, options_.min_weight);

    if (normalization != solver::SolveStatus::Success) {
        return normalization;
    }

    // Commit only after the measurement update succeeds.
    particles_.swap(candidate);

    if (options_.resample_after_update &&
        effective_sample_size() <
            options_.resample_ess_ratio *
            static_cast<double>(particles_.size())) {
        return resample();
    }

    return solver::SolveStatus::Success;
}

solver::SolveStatus ParticleFilter::normalize_weights() {
    if (!initialized_) {
        return solver::SolveStatus::NumericalFailure;
    }

    std::vector<Particle> candidate = particles_;

    const solver::SolveStatus status =
        normalize_population(candidate, options_.min_weight);

    if (status != solver::SolveStatus::Success) {
        return status;
    }

    particles_.swap(candidate);
    return solver::SolveStatus::Success;
}

solver::SolveStatus ParticleFilter::resample() {
    if (!initialized_ || particles_.empty()) {
        return solver::SolveStatus::NumericalFailure;
    }

    const std::size_t count = particles_.size();

    double total_weight = 0.0;
    for (const Particle& p : particles_) {
        if (!std::isfinite(p.weight) || p.weight < 0.0) {
            return solver::SolveStatus::NumericalFailure;
        }
        total_weight += p.weight;
    }

    if (!std::isfinite(total_weight) || total_weight <= 0.0) {
        return solver::SolveStatus::NumericalFailure;
    }

    // Systematic resampling: one random offset, then equally spaced
    // sample positions across the cumulative weight distribution.
    double u = random_.uniform01();

    if (!std::isfinite(u)) {
        return solver::SolveStatus::NonFiniteInput;
    }

    // Defend against a source that returns a value just outside [0, 1).
    u = std::max(0.0, std::min(
        u, std::nextafter(1.0, 0.0)
    ));

    const double step = 1.0 / static_cast<double>(count);
    const double start = u * step;

    std::vector<Particle> candidate;
    candidate.reserve(count);

    std::size_t source_index = 0;
    double cumulative =
        particles_[0].weight / total_weight;

    for (std::size_t i = 0; i < count; ++i) {
        const double target =
            start + static_cast<double>(i) * step;

        while (source_index + 1 < count &&
               target >= cumulative) {
            ++source_index;
            cumulative +=
                particles_[source_index].weight / total_weight;
        }

        Particle selected = particles_[source_index];
        selected.weight = step;
        candidate.push_back(std::move(selected));
    }

    // Resampling has produced equal weights; apply the configured
    // floor only when it is valid for this population.
    const solver::SolveStatus normalization =
        normalize_population(candidate, options_.min_weight);

    if (normalization != solver::SolveStatus::Success) {
        return normalization;
    }

    particles_.swap(candidate);
    return solver::SolveStatus::Success;
}

ParticleEstimate ParticleFilter::estimate() const {
    ParticleEstimate result(state_dimension_);

    if (!initialized_ || particles_.empty() ||
        state_dimension_ == 0) {
        result.status = solver::SolveStatus::NumericalFailure;
        return result;
    }

    double total_weight = 0.0;

    for (const Particle& p : particles_) {
        if (p.state.size() != state_dimension_ ||
            !finite_vector(p.state) ||
            !std::isfinite(p.weight) ||
            p.weight < 0.0) {
            result.status = solver::SolveStatus::NumericalFailure;
            return result;
        }

        total_weight += p.weight;
    }

    if (!std::isfinite(total_weight) || total_weight <= 0.0) {
        result.status = solver::SolveStatus::NumericalFailure;
        return result;
    }

    // Weighted mean.
    for (const Particle& p : particles_) {
        const double weight = p.weight / total_weight;

        for (std::size_t i = 0; i < state_dimension_; ++i) {
            result.mean(i) += weight * p.state(i);
        }
    }

    // Weighted covariance:
    // Cov[x] = sum_i w_i (x_i - mean)(x_i - mean)^T.
    for (const Particle& p : particles_) {
        const double weight = p.weight / total_weight;

        for (std::size_t i = 0; i < state_dimension_; ++i) {
            const double di = p.state(i) - result.mean(i);

            for (std::size_t j = 0; j < state_dimension_; ++j) {
                const double dj = p.state(j) - result.mean(j);
                result.covariance(i, j) += weight * di * dj;
            }
        }
    }

    result.effective_sample_size = compute_ess(particles_);

    if (!std::isfinite(result.effective_sample_size) ||
        !finite_vector(result.mean)) {
        result.status = solver::SolveStatus::NumericalFailure;
        return result;
    }

    result.status = solver::SolveStatus::Success;
    return result;
}

double ParticleFilter::effective_sample_size() const {
    if (!initialized_ || particles_.empty()) {
        return 0.0;
    }

    return compute_ess(particles_);
}

std::size_t ParticleFilter::particle_count() const {
    return particles_.size();
}

std::size_t ParticleFilter::state_dim() const {
    return initialized_ ? state_dimension_ : 0;
}

const std::vector<Particle>& ParticleFilter::particles() const {
    return particles_;
}

void ParticleFilter::reset() {
    particles_.clear();
    state_dimension_ = 0;
    initialized_ = false;
}

} // namespace mcl
} // namespace filters
} // namespace math
