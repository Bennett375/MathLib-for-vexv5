#pragma once

#include "mathlib/geometry/circle.hpp"
#include "mathlib/geometry/point.hpp"

namespace math {
/** Return the shortest signed angle difference in [-pi, pi). */
double angle_error(double target, double current);

}