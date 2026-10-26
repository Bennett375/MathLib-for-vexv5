#include "mathlib/control/slew.hpp"

namespace math {
namespace control {
    double apply_slew(double prev_inp, double commanded_inp, double dt, double slew_rate) {
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