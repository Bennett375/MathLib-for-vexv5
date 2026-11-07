#include "mathlib/geometry/geometry.hpp"
#include "mathlib/arithmetic/arithmetic.hpp"

#include <cmath>

namespace math {
angle<> angle_error(angle<> target, angle<> current) {
  double result = std::fmod(
      target.get_value() - current.get_value() + PI, TAU);
  if (result < 0.0) {
    result += TAU;
  }
  return angle<>(result - PI);
}

}
