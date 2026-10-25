#include "geometry.hpp"
#include "mathlib/arithmetic/arithmetic.hpp"

#include <cmath>

namespace math {
double angle_error(double target, double current) {
  double result = std::fmod(target - current + PI, TAU);
  if (result < 0.0) {
    result += TAU;
  }
  return result - PI;
}

}