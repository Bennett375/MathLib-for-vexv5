
#pragma once

#include "mathlib/filtering/kalman.hpp"

#include <functional>

namespace math {
namespace filters {

class ExtendedKalmanFilter : public KalmanFilter {
public:
    // Nonlinear motion model: f(x, u).
    typedef std::function<linalg::Vector<double>(
        const linalg::Vector<double>&,
        const linalg::Vector<double>&
    )> StateFunction;

    // Jacobian of f with respect to x: df/dx.
    typedef std::function<linalg::Matrix<double>(
        const linalg::Vector<double>&,
        const linalg::Vector<double>&
    )> StateJacobian;

    // Nonlinear measurement model: h(x).
    typedef std::function<linalg::Vector<double>(
        const linalg::Vector<double>&
    )> MeasurementFunction;

    // Jacobian of h with respect to x: dh/dx.
    typedef std::function<linalg::Matrix<double>(
        const linalg::Vector<double>&
    )> MeasurementJacobian;

    // Measurement residual: z - h(x), with optional custom
    // handling such as angle wrapping.
    typedef std::function<linalg::Vector<double>(
        const linalg::Vector<double>&,
        const linalg::Vector<double>&
    )> ResidualFunction;

    ExtendedKalmanFilter();

    // Nonlinear prediction with a control Vector.
    solver::SolveStatus predict_nonlinear(
        const StateFunction& state_function,
        const StateJacobian& state_jacobian,
        const linalg::Vector<double>& control,
        const linalg::Matrix<double>& Q);

    // Nonlinear prediction without a control input.
    // The model receives an empty control vector.
    solver::SolveStatus predict_nonlinear(
        const StateFunction& state_function,
        const StateJacobian& state_jacobian,
        const linalg::Matrix<double>& Q);

    // Nonlinear measurement update.
    // If residual_function is empty, uses measurement - h(x).
    solver::SolveStatus update_nonlinear(
        const linalg::Vector<double>& measurement,
        const MeasurementFunction& measurement_function,
        const MeasurementJacobian& measurement_jacobian,
        const linalg::Matrix<double>& R,
        const ResidualFunction& residual_function = ResidualFunction());
};

} // namespace filters
} // namespace math
