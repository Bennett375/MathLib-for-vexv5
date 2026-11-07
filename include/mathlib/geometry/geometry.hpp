#pragma once

#include "mathlib/geometry/circle.hpp"
#include "mathlib/geometry/point.hpp"
#include "mathlib/units/angleunit.hpp"

namespace math {
/** Return the shortest signed angle difference in [-pi, pi). */
angle<> angle_error(angle<> target, angle<> current);

}
