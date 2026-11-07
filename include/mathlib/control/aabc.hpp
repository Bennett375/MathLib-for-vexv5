#pragma once

#include "mathlib/units/angleunit.hpp"
#include "mathlib/units/numbers.hpp"
#include "mathlib/units/powerunit.hpp"

namespace math {

struct AngularControllerConfig {
    // Angular limits are in rad/s and rad/s^2; durations are in seconds.
    // Position tolerances and margins are in radians.
    real max_velocity = 6.7;
    real max_acceleration = 8.0;
    real max_braking_acceleration = 10.0;
    angle<> position_tolerance = 0.02;
    real velocity_tolerance = 0.10;
    real settling_time = 0.12;
    angle<> position_margin = 0.0;
    real velocity_response = 4.0;

    // Optional drivetrain model: angular acceleration per command unit and
    // rotational drag coefficient (1/s).
    bool use_dynamics_model = false;
    real motor_acceleration_gain = 0.0;
    real rotational_drag = 0.0;

    // Fallback velocity controller gains, in command units per radian/second.
    real velocity_command_gain = 3.0;
    real velocity_feedforward_gain = 0.0;

    // Motor command limits and optional deadband/static-friction compensation.
    powerunit<> max_command = 50.0;
    powerunit<> command_deadband = 0.0;
    powerunit<> minimum_command = 0.0;

    // Maximum accepted update timestep, in seconds.
    real max_dt = 0.05;
};

class AngularController {
public:
    explicit AngularController(const AngularControllerConfig& config);

    void set_target(angle<> target);
    void reset();

    // Heading is in radians, angular velocity in rad/s, and dt in seconds.
    // Returns a signed differential motor command in the default power unit.
    powerunit<> update(
        angle<> heading,
        real angular_velocity,
        real dt);

    bool is_settled() const;
    angle<> get_target() const;
    real get_desired_velocity() const;

private:
    AngularControllerConfig config;
    angle<> target_heading = 0.0;
    real desired_velocity = 0.0;
    real settling_elapsed = 0.0;
    bool settled = false;
    real previous_desired_velocity = 0.0;

    static double clamp(double value, double lower, double upper);
    static angle<> wrap_angle(angle<> angle);

    real calculate_desired_velocity(angle<> error) const;
    real calculate_acceleration(
        real target_velocity,
        real actual_velocity,
        real profile_acceleration) const;
    powerunit<> calculate_model_command(
        real desired_acceleration,
        real actual_velocity) const;
    powerunit<> calculate_velocity_command(
        real target_velocity,
        real actual_velocity) const;
    powerunit<> constrain_command(powerunit<> command) const;
    void update_settling(
        angle<> error,
        real angular_velocity,
        real dt);
};

}
