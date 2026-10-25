#include "mathlib/control/pid.hpp"

#include <cmath>

namespace math {
namespace control {
namespace {

// Branch-based clamp. Inlined here so the hot path makes no cross-TU calls.
inline double clamp_to(double value, double lo, double hi) {
    return value < lo ? lo : (value > hi ? hi : value);
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
    if (!std::isfinite(error) || !std::isfinite(dt) || dt <= 0.0) {
        return last_output_;
    }

    // Derivative: skipped on the first sample to avoid a derivative kick.
    double derivative = 0.0;

    if (state_.initialized) {
        const double raw = (error - state_.previous_error) / dt;
        const double alpha = options_.derivative_filter_alpha;

        // alpha >= 1 means no filtering, so skip the blend entirely.
        derivative = (alpha >= 1.0)
            ? raw
            : clamp_to(alpha, 0.0, 1.0) * raw +
              (1.0 - clamp_to(alpha, 0.0, 1.0)) * state_.filtered_derivative;
    } else {
        // Same blend as above with raw = 0.
        const double alpha = clamp_to(options_.derivative_filter_alpha, 0.0, 1.0);
        derivative = (1.0 - alpha) * state_.filtered_derivative;
    }

    // Integral with optional hard limit (0 disables the limit).
    double integral = state_.integral + error * dt;

    if (options_.integral_limit > 0.0) {
        integral = clamp_to(
            integral, -options_.integral_limit, options_.integral_limit);
    }

    // P and D do not depend on the integral, so compute them once. The sum is
    // kept in P + I + D order so results match a sequential dot product.
    const double p_term = gains_.kp * error;
    const double d_term = gains_.kd * derivative;
    double output = p_term + gains_.ki * integral + d_term;

    // Conditional-integration anti-windup.
    if ((output > options_.output_max && error > 0.0) ||
        (output < options_.output_min && error < 0.0)) {
        integral = state_.integral;
        output = p_term + gains_.ki * integral + d_term;
    }

    output = clamp_to(output, options_.output_min, options_.output_max);

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