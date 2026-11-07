#pragma once

#include <limits>

#include "mathlib/units/numbers.hpp"

namespace math {
namespace control {

struct PIDGains {
    real kp;
    real ki;
    real kd;

    PIDGains(real p = 0.0,
             real i = 0.0,
             real d = 0.0)
        : kp(p), ki(i), kd(d) {}
};

struct PIDOptions {
    real integral_limit;
    real output_min;
    real output_max;
    real derivative_filter_alpha;

    PIDOptions()
        : integral_limit(0.0),
          output_min(-std::numeric_limits<double>::max()),
          output_max(std::numeric_limits<double>::max()),
          derivative_filter_alpha(1.0) {}
};

struct PIDState {
    real integral;
    real previous_error;
    real filtered_derivative;
    bool initialized;

    PIDState()
        : integral(0.0),
          previous_error(0.0),
          filtered_derivative(0.0),
          initialized(false) {}
};

class PIDController {
public:
    PIDController(
        const PIDGains& gains,
        const PIDOptions& options = PIDOptions());

    real update(real error, real dt);
    void reset();

    const PIDGains& gains() const;
    void set_gains(const PIDGains& gains);

    const PIDState& state() const;
    real last_output() const;

private:
    PIDGains gains_;
    PIDOptions options_;
    PIDState state_;
    real last_output_;
};

} // namespace control
} // namespace math
