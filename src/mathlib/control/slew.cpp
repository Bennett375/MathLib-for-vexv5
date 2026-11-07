#include "mathlib/control/slew.hpp"

namespace math {
namespace control {
    real apply_slew(real prev_inp, real commanded_inp, real dt, real slew_rate) {
        if (prev_inp != commanded_inp) {
            // Increasing commands
            if (commanded_inp - prev_inp > slew_rate * dt) {
                return prev_inp + slew_rate * dt;
            }
            // Decreasing commands
            if (commanded_inp - prev_inp < -slew_rate * dt) {
                return prev_inp - slew_rate * dt;
            }
        }
        return commanded_inp;
    }
}
}
