#pragma once


namespace math {
namespace control {
    double apply_slew(double prev_inp, double commanded_inp, double dt, double slew_rate = 1.5);
}
}
