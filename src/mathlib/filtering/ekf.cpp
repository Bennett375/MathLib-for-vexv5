
#include "ekf.hpp"


#include <cmath>
#include <cstddef>

namespace math {
namespace filters {

ExtendedKalmanFilter::ExtendedKalmanFilter()
    : KalmanFilter() {
}

solver::SolveStatus ExtendedKalmanFilter::predict_nonlinear(
    const StateFunction& state_function,
    const StateJacobian& state_jacobian,
    const linalg::Vector<double>& control,
    const linalg::Matrix<double>& Q) {

    if (!initialized()) {
        return solver::SolveStatus::InvalidDimensions;
    }

    if (!state_function || !state_jacobian) {
        return solver::SolveStatus::InvalidDimensions;
    }

    const linalg::Vector<double>& x = state();
    const std::size_t n = state_dim();

    if (n == 0 ||
        Q.rows() != n ||
        Q.cols() != n) {
        return solver::SolveStatus::InvalidDimensions;
    }

    if (!solver::detail::vector_is_finite(x) ||
        !solver::detail::vector_is_finite(control) ||
        !solver::detail::matrix_is_finite(Q)) {
        return solver::SolveStatus::NonFiniteInput;
    }

    // Evaluate the nonlinear motion model and its Jacobian
    // at the current estimated state.
    const linalg::Vector<double> predicted_mean =
        state_function(x, control);

    const linalg::Matrix<double> F =
        state_jacobian(x, control);

    if (predicted_mean.size() != n ||
        F.rows() != n ||
        F.cols() != n) {
        return solver::SolveStatus::InvalidDimensions;
    }

    if (!solver::detail::vector_is_finite(predicted_mean) ||
        !solver::detail::matrix_is_finite(F)) {
        return solver::SolveStatus::NumericalFailure;
    }

    // Convert the nonlinear predicted mean into an equivalent
    // affine prediction:
    //
    // predicted_mean = F*x + offset
    //
    // The base filter then propagates covariance using F.
    const linalg::Vector<double> linearized_mean = F * x;
    const linalg::Vector<double> offset =
        predicted_mean - linearized_mean;

    if (!solver::detail::vector_is_finite(offset)) {
        return solver::SolveStatus::NumericalFailure;
    }

    const linalg::Matrix<double> B = linalg::Matrix<double>::identity(n);

    return KalmanFilter::predict(F, B, offset, Q);
}

solver::SolveStatus ExtendedKalmanFilter::predict_nonlinear(
    const StateFunction& state_function,
    const StateJacobian& state_jacobian,
    const linalg::Matrix<double>& Q) {

    const linalg::Vector<double> no_control(0);

    return predict_nonlinear(
        state_function,
        state_jacobian,
        no_control,
        Q);
}

solver::SolveStatus ExtendedKalmanFilter::update_nonlinear(
    const linalg::Vector<double>& measurement,
    const MeasurementFunction& measurement_function,
    const MeasurementJacobian& measurement_jacobian,
    const linalg::Matrix<double>& R,
    const ResidualFunction& residual_function) {

    if (!initialized()) {
        return solver::SolveStatus::InvalidDimensions;
    }

    if (!measurement_function || !measurement_jacobian) {
        return solver::SolveStatus::InvalidDimensions;
    }

    const linalg::Vector<double>& x = state();
    const std::size_t n = state_dim();

    if (n == 0 || measurement.size() == 0) {
        return solver::SolveStatus::InvalidDimensions;
    }

    if (!solver::detail::vector_is_finite(x) ||
        !solver::detail::vector_is_finite(measurement) ||
        !solver::detail::matrix_is_finite(R)) {
        return solver::SolveStatus::NonFiniteInput;
    }

    // Evaluate h(x) and H = dh/dx at the current estimate.
    const linalg::Vector<double> predicted_measurement =
        measurement_function(x);

    const linalg::Matrix<double> H =
        measurement_jacobian(x);

    const std::size_t m = measurement.size();

    if (predicted_measurement.size() != m ||
        H.rows() != m ||
        H.cols() != n ||
        R.rows() != m ||
        R.cols() != m) {
        return solver::SolveStatus::InvalidDimensions;
    }

    if (!solver::detail::vector_is_finite(predicted_measurement) ||
        !solver::detail::matrix_is_finite(H)) {
        return solver::SolveStatus::NumericalFailure;
    }

    // Calculate the innovation using either the supplied residual
    // function or the standard measurement-minus-prediction rule.
    linalg::Vector<double> innovation = residual_function
        ? residual_function(measurement, predicted_measurement)
        : measurement - predicted_measurement;

    if (innovation.size() != m) {
        return solver::SolveStatus::InvalidDimensions;
    }

    if (!solver::detail::vector_is_finite(innovation)) {
        return solver::SolveStatus::NumericalFailure;
    }

    // The base filter calculates:
    //
    // adjusted_measurement - H*x
    //
    // Choose adjusted_measurement = innovation + H*x, making
    // the base filter use the nonlinear model's innovation.
    const linalg::Vector<double> Hx = H * x;
    const linalg::Vector<double> adjusted_measurement = innovation + Hx;

    if (!solver::detail::vector_is_finite(adjusted_measurement)) {
        return solver::SolveStatus::NumericalFailure;
    }

    return KalmanFilter::update(
        adjusted_measurement,
        H,
        R);
}

} // namespace filters
} // namespace math
