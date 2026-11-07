#include "mathlib/arithmetic/arithmetic.hpp"

#include <cmath>

namespace math {

real mean(const std::vector<double>& values) {
  if (values.empty()) {
    return real(0.0);
  }
  double total = 0.0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    total += values[i];
  }
  return real(total / static_cast<double>(values.size()));
}

real mean(real first, real second) {
  return real((first.get_value() + second.get_value()) / 2.0);
}

int sign(int value) {
  return (value > 0) - (value < 0);
}

real sign(double value) {
  return real((value > 0.0) - (value < 0.0));
}

real sign(real value) {
  return sign(value.get_value());
}

real clamp(real value, real minimum, real maximum) {
  return real(std::max(minimum.get_value(),
                       std::min(maximum.get_value(), value.get_value())));
}

real power(real base, real exponent) {
  return real(std::pow(base.get_value(), exponent.get_value()));
}

real exp(real exponent) {
  return power(e, exponent);
}

real sqr(real value) {
  const double raw = value.get_value();
  return real(raw * raw);
}

real sqrt(real value) {
  return real(std::sqrt(value.get_value()));
}

}
