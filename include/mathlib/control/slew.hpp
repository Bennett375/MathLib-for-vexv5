#pragma once

#include "mathlib/units/numbers.hpp"


namespace math {
namespace control {
    real apply_slew(real prev_inp, real commanded_inp, real dt, real slew_rate = 1.5);
}
}
