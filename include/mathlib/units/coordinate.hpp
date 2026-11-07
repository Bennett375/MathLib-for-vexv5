#pragma once

#include "mathlib/units/numbers.hpp"

namespace math {

enum FieldUnits {
    INCHES,
    CM,
    MM,
    FEET,
};

constexpr FieldUnits default_field_unit_type = FieldUnits::INCHES;

template <FieldUnits unit = default_field_unit_type>
struct fieldunit : public real {
    constexpr fieldunit() noexcept = default;

    constexpr fieldunit(double value) noexcept
        : real(value) {}

    // Implicit conversion from real to a Field Unit
    constexpr fieldunit(const real& value) noexcept
        : real(value.get_value()) {}

    // Convert between field units without requiring an explicit .as<>() call.
    template <FieldUnits other>
    constexpr fieldunit(const fieldunit<other>& value) noexcept
        : real(value.get_value() * meters(other) / meters(unit)) {}

    // Explicit unit conversion: inches.as<fieldunit<CM>>().
    template <typename T>
    T as() const {
        return conversion<fieldunit<unit>, T>::convert(*this);
    }

private:
    static constexpr double meters(FieldUnits value) noexcept {
        return value == FieldUnits::INCHES
            ? 0.0254
            : (value == FieldUnits::CM
                ? 0.01
                : (value == FieldUnits::MM ? 0.001 : 0.3048));
    }
};

template <FieldUnits from, FieldUnits to>
struct conversion<fieldunit<from>, fieldunit<to>> {
    static constexpr fieldunit<to> convert(
        const fieldunit<from>& value) noexcept {
        return fieldunit<to>(value);
    }
};

template <FieldUnits unit>
struct conversion<fieldunit<unit>, double> {
    static constexpr double convert(
        const fieldunit<unit>& value) noexcept {
        return value.get_value();
    }
};

template <FieldUnits unit>
struct conversion<fieldunit<unit>, float> {
    static constexpr float convert(
        const fieldunit<unit>& value) noexcept {
        return static_cast<float>(value.get_value());
    }
};

template <FieldUnits unit>
struct conversion<fieldunit<unit>, int> {
    static constexpr int convert(
        const fieldunit<unit>& value) noexcept {
        return static_cast<int>(value.get_value());
    }
};

template <FieldUnits unit>
struct conversion<fieldunit<unit>, real> {
    static constexpr real convert(
        const fieldunit<unit>& value) noexcept {
        return real(value.get_value());
    }
};

template <>
struct conversion<fieldunit<FieldUnits::INCHES>, fieldunit<FieldUnits::CM>> {
    static fieldunit<FieldUnits::CM> convert(const fieldunit<FieldUnits::INCHES>& fu) {
        return fu * 2.54;
    }
};

template <>
struct conversion<fieldunit<FieldUnits::CM>, fieldunit<FieldUnits::INCHES>> {
    static fieldunit<FieldUnits::INCHES> convert(const fieldunit<FieldUnits::CM>& fu) {
        return fu / 2.54;
    }
};

template <>
struct conversion<fieldunit<FieldUnits::CM>, fieldunit<FieldUnits::MM>> {
    static fieldunit<FieldUnits::MM> convert(const fieldunit<FieldUnits::CM>& fu) {
        return fu * 10.0;
    }
};

template <>
struct conversion<fieldunit<FieldUnits::MM>, fieldunit<FieldUnits::CM>> {
    static fieldunit<FieldUnits::CM> convert(const fieldunit<FieldUnits::MM>& fu) {
        return fu * 0.1;
    }
};

template <>
struct conversion<fieldunit<FieldUnits::INCHES>, fieldunit<FieldUnits::FEET>> {
    static fieldunit<FieldUnits::FEET> convert(const fieldunit<FieldUnits::INCHES>& fu) {
        return fu / 12.0;
    }
};

template <>
struct conversion<fieldunit<FieldUnits::FEET>, fieldunit<FieldUnits::INCHES>> {
    static fieldunit<FieldUnits::INCHES> convert(const fieldunit<FieldUnits::FEET>& fu) {
        return fu * 12.0;
    }
};

} // namespace math
