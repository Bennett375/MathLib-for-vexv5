#include "arithmetic.hpp"

#include <cmath>

namespace math {

double mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  double total = 0.0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    total += values[i];
  }
  return total / static_cast<double>(values.size());
}

double mean(double first, double second) {
  return (first + second) / 2.0;
}

int sign(int value) {
  return (value > 0) - (value < 0);
}

double sign(double value) {
  return (value > 0.0) - (value < 0.0);
}

double clamp(double value, double minimum, double maximum) {
  return std::max(minimum, std::min(maximum, value));
}

double power(double base, double exponent) {
  return std::pow(base, exponent);
}

double exp(double exponent) {
  return power(e, exponent);
}

double sqr(double value) {
  return value * value;
}

double sqrt(double value) {
  return std::sqrt(value);
}

}