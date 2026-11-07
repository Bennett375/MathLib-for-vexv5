#include "mathlib/control/aabc.hpp"

#include <algorithm>
#include <cmath>

namespace math {

AngularController::AngularController(const AngularControllerConfig& config)
    : config(config) {
    reset();
}

void AngularController::set_target(angle<> target) {
    target_heading = wrap_angle(target);
    settled = false;
    settling_elapsed = 0.0;
    desired_velocity = 0.0;
}

void AngularController::reset() {
    desired_velocity = 0.0;
    settling_elapsed = 0.0;
    previous_desired_velocity = 0.0;
    target_heading = 0.0;
    settled = false;
}

powerunit<> AngularController::update(
    angle<> heading,
    real angular_velocity,
    real dt) {
    if (!std::isfinite(heading.get_value()) ||
        !std::isfinite(angular_velocity.get_value()) ||
        !std::isfinite(dt.get_value()) ||
        dt <= 0.0) {
        return powerunit<>(0.0);
    }

    dt = real(std::min(dt.get_value(), config.max_dt.get_value()));

    if (settled) {
        return powerunit<>(0.0);
    }

    const angle<> error = wrap_angle(
        angle<>(target_heading.get_value() - heading.get_value()));

    previous_desired_velocity = desired_velocity;
    desired_velocity = calculate_desired_velocity(error);

    const real profile_acceleration =
        (desired_velocity - previous_desired_velocity) / dt;
    const real acceleration = calculate_acceleration(
        desired_velocity,
        angular_velocity,
        profile_acceleration);

    powerunit<> command;
    if (config.use_dynamics_model &&
        config.motor_acceleration_gain.get_value() > 0.0) {
        command = calculate_model_command(acceleration, angular_velocity);
    } else {
        command = calculate_velocity_command(
            desired_velocity,
            angular_velocity);
    }

    command = constrain_command(command);
    update_settling(error, angular_velocity, dt);

    if (settled) {
        return powerunit<>(0.0);
    }

    return command;
}

bool AngularController::is_settled() const {
    return settled;
}

angle<> AngularController::get_target() const {
    return target_heading;
}

real AngularController::get_desired_velocity() const {
    return desired_velocity;
}

double AngularController::clamp(
    double value,
    double lower,
    double upper) {
    return std::max(lower, std::min(value, upper));
}

angle<> AngularController::wrap_angle(angle<> angle_value) {
    const double pi = 3.14159265358979323846;
    const double tau = 2.0 * pi;

    double value = std::fmod(angle_value.get_value() + pi, tau);

    if (value < 0.0) {
        value += tau;
    }

    return angle<>(value - pi);
}

real AngularController::calculate_desired_velocity(angle<> error) const {
    const double distance = std::max(
        0.0,
        std::abs(error.get_value()) - config.position_margin.get_value());

    if (distance <= config.position_tolerance.get_value()) {
        return real(0.0);
    }

    double velocity_limit = config.max_velocity.get_value();

    if (config.max_braking_acceleration.get_value() > 0.0) {
        velocity_limit = std::sqrt(
            2.0 * config.max_braking_acceleration.get_value() * distance);
    }

    velocity_limit = std::min(
        velocity_limit,
        config.max_velocity.get_value());

    return real(std::copysign(velocity_limit, error.get_value()));
}

real AngularController::calculate_acceleration(
    real target_velocity,
    real actual_velocity,
    real profile_acceleration) const {
    const double velocity_error =
        target_velocity.get_value() - actual_velocity.get_value();
    double acceleration =
        profile_acceleration.get_value() +
        config.velocity_response.get_value() * velocity_error;

    const bool braking = acceleration * actual_velocity.get_value() < 0.0;
    const double limit = braking
        ? config.max_braking_acceleration.get_value()
        : config.max_acceleration.get_value();

    return real(clamp(acceleration, -limit, limit));
}

powerunit<> AngularController::calculate_model_command(
    real desired_acceleration,
    real actual_velocity) const {
    return powerunit<>((
        desired_acceleration.get_value() +
        config.rotational_drag.get_value() * actual_velocity.get_value()
    ) / config.motor_acceleration_gain.get_value());
}

powerunit<> AngularController::calculate_velocity_command(
    real target_velocity,
    real actual_velocity) const {
    const double velocity_error =
        target_velocity.get_value() - actual_velocity.get_value();
    const double feedforward =
        config.velocity_feedforward_gain.get_value() *
        target_velocity.get_value();
    const double feedback =
        config.velocity_command_gain.get_value() * velocity_error;

    return powerunit<>(feedforward + feedback);
}

powerunit<> AngularController::constrain_command(powerunit<> command) const {
    double value = clamp(
        command.get_value(),
        -config.max_command.get_value(),
        config.max_command.get_value());

    const double magnitude = std::abs(value);

    if (magnitude <= config.command_deadband.get_value()) {
        return powerunit<>(0.0);
    }

    if (config.minimum_command.get_value() > 0.0 &&
        magnitude < config.minimum_command.get_value()) {
        value = std::copysign(
            config.minimum_command.get_value(),
            value);
    }

    return powerunit<>(clamp(
        value,
        -config.max_command.get_value(),
        config.max_command.get_value()));
}

void AngularController::update_settling(
    angle<> error,
    real angular_velocity,
    real dt) {
    const bool position_ok =
        std::abs(error.get_value()) <=
        config.position_tolerance.get_value();
    const bool velocity_ok =
        std::abs(angular_velocity.get_value()) <=
        config.velocity_tolerance.get_value();

    if (position_ok && velocity_ok) {
        settling_elapsed += dt;
    } else {
        settling_elapsed = 0.0;
    }

    if (settling_elapsed.get_value() >= config.settling_time.get_value()) {
        settled = true;
    }
}

}