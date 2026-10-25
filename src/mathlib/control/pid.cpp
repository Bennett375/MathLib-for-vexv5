#include "mathlib/control/pid.hpp"

#include "mathlib/arithmetic/arithmetic.hpp"
#include "mathlib/linalg/vector.hpp"

#include <cmath>

namespace math {
namespace control {
namespace {

linalg::Vector<double> gain_vector(const PIDGains& gains) {
    return linalg::Vector<double>{gains.kp, gains.ki, gains.kd};
}

linalg::Vector<double> term_vector(
    double error,
    double integral,
    double derivative) {

    return linalg::Vector<double>{error, integral, derivative};
}

double combine(
    const PIDGains& gains,
    double error,
    double integral,
    double derivative) {

    return linalg::dot(
        gain_vector(gains),
        term_vector(error, integral, derivative));
}

bool finite(double value) {
    return std::isfinite(value);
}

double clamp_symmetric(double value, double limit) {
    return math::clamp(value, -limit, limit);
}

} // namespace

PIDController::PIDController(
    const PIDGains& gains,
    const PIDOptions& options)
    : gains_(gains),
      options_(options),
      state_(),
      last_output_(0.0) {}

double PIDController::update(double error, double dt) {
    // Reject bad samples without disturbing the controller state.
    if (!finite(error) || !finite(dt) || dt <= 0.0) {
        return last_output_;
    }

    // A filter alpha outside [0, 1] is meaningless, so clamp it.
    const double alpha =
        math::clamp(options_.derivative_filter_alpha, 0.0, 1.0);

    // Derivative: skip the first sample, since there is no previous error and
    // computing one would produce a derivative kick.
    double raw_derivative = 0.0;

    if (state_.initialized) {
        raw_derivative = (error - state_.previous_error) / dt;
    }

    // Low-pass filter: alpha = 1 disables filtering.
    const double derivative =
        alpha * raw_derivative +
        (1.0 - alpha) * state_.filtered_derivative;

    // Integral with optional hard limit (0 disables the limit).
    double integral = state_.integral + error * dt;

    if (options_.integral_limit > 0.0) {
        integral = clamp_symmetric(integral, options_.integral_limit);
    }

    // Unsaturated output using the candidate integral.
    double output = combine(gains_, error, integral, derivative);

    // Conditional-integration anti-windup: if the output is saturated and the
    // error would drive it further into saturation, keep the old integral.
    const bool saturated_high =
        output > options_.output_max && error > 0.0;
    const bool saturated_low =
        output < options_.output_min && error < 0.0;

    if (saturated_high || saturated_low) {
        integral = state_.integral;
        output = combine(gains_, error, integral, derivative);
    }

    output = math::clamp(
        output, options_.output_min, options_.output_max);

    state_.integral = integral;
    state_.previous_error = error;
    state_.filtered_derivative = derivative;
    state_.initialized = true;
    last_output_ = output;

    return output;
}

void PIDController::reset() {
    state_ = PIDState();
    last_output_ = 0.0;
}

const PIDGains& PIDController::gains() const {
    return gains_;
}

void PIDController::set_gains(const PIDGains& gains) {
    gains_ = gains;
}

const PIDState& PIDController::state() const {
    return state_;
}

double PIDController::last_output() const {
    return last_output_;
}

} // namespace control
} // namespace math