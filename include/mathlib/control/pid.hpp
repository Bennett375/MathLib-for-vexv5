#pragma once

#include <limits>

namespace math {
namespace control {

struct PIDGains {
    double kp;
    double ki;
    double kd;

    PIDGains(double p = 0.0,
             double i = 0.0,
             double d = 0.0)
        : kp(p), ki(i), kd(d) {}
};

struct PIDOptions {
    double integral_limit;
    double output_min;
    double output_max;
    double derivative_filter_alpha;

    PIDOptions()
        : integral_limit(0.0),
          output_min(-std::numeric_limits<double>::max()),
          output_max(std::numeric_limits<double>::max()),
          derivative_filter_alpha(1.0) {}
};

struct PIDState {
    double integral;
    double previous_error;
    double filtered_derivative;
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

    double update(double error, double dt);
    void reset();

    const PIDGains& gains() const;
    void set_gains(const PIDGains& gains);

    const PIDState& state() const;
    double last_output() const;

private:
    PIDGains gains_;
    PIDOptions options_;
    PIDState state_;
    double last_output_;
};

} // namespace control
} // namespace math