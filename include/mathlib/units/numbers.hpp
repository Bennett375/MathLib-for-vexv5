#pragma once

#include "mathlib/abstract/object.hpp"

#include <type_traits>

namespace math {

/** Constant pi. */
constexpr double PI = 3.14159265358979323846;
/** Constant 2*pi. */
constexpr double TAU = 2.0 * PI;

constexpr double e = 2.718281828459045;

class real : public MathObj<real> {
private:
    double value{0.0};

public:
    // Default Constructor
    constexpr real() noexcept = default;

    // Constructors. These are intentionally implicit so existing code that
    // passes C++ primitives keeps working with APIs that use math::real.
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    constexpr real(T num) noexcept : value(static_cast<double>(num)) {}

    // Getter
    [[nodiscard]] constexpr double get_value() const noexcept { 
        return value; 
    }

    // Conversion Operators
    [[nodiscard]] constexpr operator double() const noexcept { 
        return value; 
    }
    [[nodiscard]] constexpr explicit operator float() const noexcept { 
        return static_cast<float>(value); 
    }
    [[nodiscard]] constexpr explicit operator char() const noexcept { 
        return static_cast<char>(value); 
    }

    // Increment and Decrement Operators
    constexpr real& operator++() noexcept { 
        ++value; 
        return *this; 
    }
    constexpr real operator++(int) noexcept { 
        real temp = *this; 
        ++value; 
        return temp; 
    }
    constexpr real& operator--() noexcept { 
        --value; 
        return *this; 
    }
    constexpr real operator--(int) noexcept { 
        real temp = *this; 
        --value; 
        return temp; 
    }

    // Arithmetic Operators (real + real)
    [[nodiscard]] constexpr real operator+(const real& other) const noexcept { 
        return real(value + other.value); 
    }
    [[nodiscard]] constexpr real operator-(const real& other) const noexcept { 
        return real(value - other.value); 
    }
    [[nodiscard]] constexpr real operator*(const real& other) const noexcept { 
        return real(value * other.value); 
    }
    [[nodiscard]] constexpr real operator/(const real& other) const noexcept { 
        return real(value / other.value); 
    }

    // Unary Operators
    [[nodiscard]] constexpr real operator+() const noexcept { 
        return *this; 
    }
    [[nodiscard]] constexpr real operator-() const noexcept { 
        return real(-value); 
    }

    // Compound Assignment Operators
    constexpr real& operator+=(const real& other) noexcept { 
        value += other.value; 
        return *this; 
    }
    constexpr real& operator-=(const real& other) noexcept { 
        value -= other.value; 
        return *this; 
    }
    constexpr real& operator*=(const real& other) noexcept { 
        value *= other.value; 
        return *this; 
    }
    constexpr real& operator/=(const real& other) noexcept { 
        value /= other.value; 
        return *this; 
    }

    // Comparison Operators
    [[nodiscard]] constexpr bool operator==(const real& other) const noexcept { 
        return value == other.value; 
    }
    [[nodiscard]] constexpr bool operator!=(const real& other) const noexcept { 
        return value != other.value; 
    }
    [[nodiscard]] constexpr bool operator<(const real& other) const noexcept { 
        return value < other.value; 
    }
    [[nodiscard]] constexpr bool operator>(const real& other) const noexcept { 
        return value > other.value; 
    }
    [[nodiscard]] constexpr bool operator<=(const real& other) const noexcept { 
        return value <= other.value; 
    }
    [[nodiscard]] constexpr bool operator>=(const real& other) const noexcept { 
        return value >= other.value; 
    }

    // Mixed-Type Arithmetic (real + primitive). Templates keep these exact
    // matches for int, float, and double so the built-in candidates cannot
    // make an otherwise valid primitive expression ambiguous.
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] constexpr real operator+(T num) const noexcept { 
        return real(value + static_cast<double>(num)); 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] constexpr real operator-(T num) const noexcept { 
        return real(value - static_cast<double>(num)); 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] constexpr real operator*(T num) const noexcept { 
        return real(value * static_cast<double>(num)); 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] constexpr real operator/(T num) const noexcept { 
        return real(value / static_cast<double>(num)); 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    constexpr real& operator+=(T num) noexcept { 
        value += static_cast<double>(num); 
        return *this; 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    constexpr real& operator-=(T num) noexcept { 
        value -= static_cast<double>(num); 
        return *this; 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    constexpr real& operator*=(T num) noexcept { 
        value *= static_cast<double>(num); 
        return *this; 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    constexpr real& operator/=(T num) noexcept { 
        value /= static_cast<double>(num); 
        return *this; 
    }

    // Mixed-Type Comparison (real == primitive)
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] constexpr bool operator==(T num) const noexcept { 
        return value == static_cast<double>(num); 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] constexpr bool operator!=(T num) const noexcept { 
        return value != static_cast<double>(num); 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] constexpr bool operator<(T num) const noexcept { 
        return value < static_cast<double>(num); 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] constexpr bool operator>(T num) const noexcept { 
        return value > static_cast<double>(num); 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] constexpr bool operator<=(T num) const noexcept { 
        return value <= static_cast<double>(num); 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] constexpr bool operator>=(T num) const noexcept { 
        return value >= static_cast<double>(num); 
    }

    // Reverse Mixed-Type Operations (primitive + real)
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] friend constexpr real operator+(T num, const real& r) noexcept { 
        return real(static_cast<double>(num) + r.value); 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] friend constexpr real operator-(T num, const real& r) noexcept { 
        return real(static_cast<double>(num) - r.value); 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] friend constexpr real operator*(T num, const real& r) noexcept { 
        return real(static_cast<double>(num) * r.value); 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] friend constexpr real operator/(T num, const real& r) noexcept { 
        return real(static_cast<double>(num) / r.value); 
    }

    // Reverse Mixed-Type Comparisons (primitive == real)
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] friend constexpr bool operator==(T num, const real& r) noexcept { 
        return static_cast<double>(num) == r.value; 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] friend constexpr bool operator!=(T num, const real& r) noexcept { 
        return static_cast<double>(num) != r.value; 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] friend constexpr bool operator<(T num, const real& r) noexcept { 
        return static_cast<double>(num) < r.value; 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] friend constexpr bool operator>(T num, const real& r) noexcept { 
        return static_cast<double>(num) > r.value; 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] friend constexpr bool operator<=(T num, const real& r) noexcept { 
        return static_cast<double>(num) <= r.value; 
    }
    template <typename T,
              typename std::enable_if<std::is_arithmetic<T>::value,
                                      int>::type = 0>
    [[nodiscard]] friend constexpr bool operator>=(T num, const real& r) noexcept { 
        return static_cast<double>(num) >= r.value; 
    }
};

// Conversions from the wrapper back to primitive scalar types. These keep
// MathObj::as<T>() usable for math::real and for unit types derived from it.
template <>
struct conversion<real, double> {
    static constexpr double convert(const real& value) noexcept {
        return value.get_value();
    }
};

template <>
struct conversion<real, float> {
    static constexpr float convert(const real& value) noexcept {
        return static_cast<float>(value.get_value());
    }
};

template <>
struct conversion<real, int> {
    static constexpr int convert(const real& value) noexcept {
        return static_cast<int>(value.get_value());
    }
};

} // namespace math
