#pragma once

#include "mathlib/units/numbers.hpp"

namespace math {

enum PowerUnits {
    PERCENT,      // percent of full speed
    RPM,          // motor output speed
    RAW_VOLTAGE,  // PROS motor.move(): -127..127
    MILLIVOLTS,   // PROS motor.move_voltage(): -12000..12000
    VOLTS         // command scale: -12.0..12.0
};

constexpr PowerUnits default_power_unit_type = PowerUnits::PERCENT;

/**
 * Default full-speed rating used for percent <-> RPM conversion. 600 RPM is
 * the blue 6:1 cartridge; the stock V5 Smart Motor ships with the green
 * 200 RPM cartridge, so pass 200 as max_rpm for a stock motor.
 */
constexpr int default_power_max_rpm = 600;

/**
 * Full-scale value for the raw motor command. PROS documents motor.move()
 * as accepting -127..127.
 */
constexpr int default_power_max_raw_voltage = 127;

/**
 * Full-scale value for the PROS motor.move_voltage() command, in millivolts.
 * This is the documented 12 V command scale, not the unregulated Smart Port
 * rail: V5 Smart Port power is battery voltage with a 12.8 V nominal rating.
 */
constexpr int default_power_max_millivolts = 12000;

/**
 * Motor power, expressed as a percentage of maximum speed, RPM, or voltage.
 *
 * The max_rpm parameter describes the full-speed rating of the motor or gear
 * cartridge (100, 200, and 600 for VEX V5 red, green, and blue cartridges).
 * The max_millivolts parameter describes the PROS voltage-command scale; it
 * is not the Smart Port supply rail.
 *
 * RAW_VOLTAGE is the raw -127..127 command accepted by PROS motor.move().
 * MILLIVOLTS and VOLTS are the voltage scales accepted by move_voltage().
 * They describe commanded voltage, not a measurement of the battery/Smart
 * Port rail.
 *
 * Conversions between different units or ratings preserve the physical
 * output rather than the raw number.
 */
template <PowerUnits unit = default_power_unit_type,
          int max_rpm = default_power_max_rpm,
          int max_millivolts = default_power_max_millivolts>
struct powerunit : public real {
    static_assert(max_rpm > 0,
                  "powerunit requires a positive maximum RPM");
    static_assert(max_millivolts > 0,
                  "powerunit requires a positive maximum voltage");

    constexpr powerunit() noexcept = default;

    constexpr powerunit(double value) noexcept
        : real(value) {}

    // Implicit conversion from real to a power unit.
    constexpr powerunit(const real& value) noexcept
        : real(value.get_value()) {}

    // Convert between units and full-scale ratings.
    template <PowerUnits other_unit, int other_max_rpm,
              int other_max_millivolts>
    constexpr powerunit(
        const powerunit<other_unit, other_max_rpm, other_max_millivolts>&
            value) noexcept
        : real(from_rpm(
              to_rpm(value.get_value(), other_unit, other_max_rpm,
                     other_max_millivolts),
              unit, max_rpm, max_millivolts)) {}

    // Explicit unit conversion: percent.as<powerunit<RPM>>().
    template <typename T>
    T as() const {
        return conversion<powerunit<unit, max_rpm, max_millivolts>,
                          T>::convert(*this);
    }

    /** Return the full-speed rating used by this unit. */
    [[nodiscard]] static constexpr int max_speed_rpm() noexcept {
        return max_rpm;
    }

    /** Return the full-scale voltage command used by this unit, in mV. */
    [[nodiscard]] static constexpr int max_voltage_millivolts() noexcept {
        return max_millivolts;
    }

    /** Return the full-scale raw command used by this unit. */
    [[nodiscard]] static constexpr int max_raw_voltage() noexcept {
        return default_power_max_raw_voltage;
    }

private:
    // RPM is the canonical unit. Percent, raw voltage, and electrical voltage
    // all describe a fraction of the same full-scale output, so converting
    // through RPM preserves the physical speed when ratings differ.
    static constexpr double to_rpm(double value, PowerUnits from,
                                   int from_max_rpm,
                                   int from_max_millivolts) noexcept {
        switch (from) {
            case PowerUnits::PERCENT:
                return value * static_cast<double>(from_max_rpm) / 100.0;
            case PowerUnits::RPM:
                return value;
            case PowerUnits::RAW_VOLTAGE:
                return value * static_cast<double>(from_max_rpm) /
                       static_cast<double>(default_power_max_raw_voltage);
            case PowerUnits::MILLIVOLTS:
                return value * static_cast<double>(from_max_rpm) /
                       static_cast<double>(from_max_millivolts);
            case PowerUnits::VOLTS:
                return value * 1000.0 * static_cast<double>(from_max_rpm) /
                       static_cast<double>(from_max_millivolts);
        }
        return 0.0;
    }

    static constexpr double from_rpm(double rpm, PowerUnits to,
                                     int to_max_rpm,
                                     int to_max_millivolts) noexcept {
        switch (to) {
            case PowerUnits::PERCENT:
                return rpm * 100.0 / static_cast<double>(to_max_rpm);
            case PowerUnits::RPM:
                return rpm;
            case PowerUnits::RAW_VOLTAGE:
                return rpm * static_cast<double>(default_power_max_raw_voltage) /
                       static_cast<double>(to_max_rpm);
            case PowerUnits::MILLIVOLTS:
                return rpm * static_cast<double>(to_max_millivolts) /
                       static_cast<double>(to_max_rpm);
            case PowerUnits::VOLTS:
                return rpm * static_cast<double>(to_max_millivolts) /
                       (1000.0 * static_cast<double>(to_max_rpm));
        }
        return 0.0;
    }
};

template <PowerUnits from, int from_max_rpm, int from_max_millivolts,
          PowerUnits to, int to_max_rpm, int to_max_millivolts>
struct conversion<powerunit<from, from_max_rpm, from_max_millivolts>,
                  powerunit<to, to_max_rpm, to_max_millivolts>> {
    static constexpr powerunit<to, to_max_rpm, to_max_millivolts> convert(
        const powerunit<from, from_max_rpm, from_max_millivolts>&
            value) noexcept {
        return powerunit<to, to_max_rpm, to_max_millivolts>(value);
    }
};

template <PowerUnits unit, int max_rpm, int max_millivolts>
struct conversion<powerunit<unit, max_rpm, max_millivolts>, double> {
    static constexpr double convert(
        const powerunit<unit, max_rpm, max_millivolts>& value) noexcept {
        return value.get_value();
    }
};

template <PowerUnits unit, int max_rpm, int max_millivolts>
struct conversion<powerunit<unit, max_rpm, max_millivolts>, float> {
    static constexpr float convert(
        const powerunit<unit, max_rpm, max_millivolts>& value) noexcept {
        return static_cast<float>(value.get_value());
    }
};

template <PowerUnits unit, int max_rpm, int max_millivolts>
struct conversion<powerunit<unit, max_rpm, max_millivolts>, int> {
    static constexpr int convert(
        const powerunit<unit, max_rpm, max_millivolts>& value) noexcept {
        return static_cast<int>(value.get_value());
    }
};

template <PowerUnits unit, int max_rpm, int max_millivolts>
struct conversion<powerunit<unit, max_rpm, max_millivolts>, real> {
    static constexpr real convert(
        const powerunit<unit, max_rpm, max_millivolts>& value) noexcept {
        return real(value.get_value());
    }
};

} // namespace math
