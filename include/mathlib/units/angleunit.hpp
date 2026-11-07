#pragma once

#include "mathlib/units/numbers.hpp"

namespace math {
enum AngleUnits {
    DEGREES,
    RADIANS
};

constexpr AngleUnits default_angle_unit_type = AngleUnits::RADIANS;

template <AngleUnits unit = default_angle_unit_type>
struct angle : public real {
    constexpr angle() noexcept = default;

    constexpr angle(double value) noexcept
        : real(value) {}

    // Implicit conversion from real to angle
    constexpr angle(const real& value) noexcept
        : real(value.get_value()) {}

    template <AngleUnits other>
    constexpr angle(const angle<other>& value) noexcept
        : real(convert_value(value.get_value(), other)) {}

    // Explicit unit conversion: degrees.as<angle<RADIANS>>().
    template <typename T>
    T as() const {
        return conversion<angle<unit>, T>::convert(*this);
    }

private:
    static constexpr double convert_value(double value,
                                          AngleUnits from) noexcept {
        return from == unit
            ? value
            : (unit == AngleUnits::RADIANS
                ? value * PI / 180.0
                : value * 180.0 / PI);
    }
};

template <AngleUnits from, AngleUnits to>
struct conversion<angle<from>, angle<to>> {
    static constexpr angle<to> convert(const angle<from>& ang) {
        return angle<to>(ang);
    }
};

template <AngleUnits unit>
struct conversion<angle<unit>, double> {
    static constexpr double convert(const angle<unit>& ang) noexcept {
        return ang.get_value();
    }
};

template <AngleUnits unit>
struct conversion<angle<unit>, float> {
    static constexpr float convert(const angle<unit>& ang) noexcept {
        return static_cast<float>(ang.get_value());
    }
};

template <AngleUnits unit>
struct conversion<angle<unit>, int> {
    static constexpr int convert(const angle<unit>& ang) noexcept {
        return static_cast<int>(ang.get_value());
    }
};

template <AngleUnits unit>
struct conversion<angle<unit>, real> {
    static constexpr real convert(const angle<unit>& ang) noexcept {
        return real(ang.get_value());
    }
};

template <>
struct conversion<angle<AngleUnits::DEGREES>, angle<AngleUnits::RADIANS>> {
    static angle<AngleUnits::RADIANS> convert(const angle<AngleUnits::DEGREES>& ang) {
        return ang * math::PI / 180.0;
    }
};

template <>
struct conversion<angle<AngleUnits::RADIANS>, angle<AngleUnits::DEGREES>> {
    static angle<AngleUnits::DEGREES> convert(const angle<AngleUnits::RADIANS>& ang) {
        return ang * 180.0 / math::PI;
    }
};
} // namespace math
