#pragma once

#include "mathlib/abstract/object.hpp"
#include "mathlib/linalg/vector.hpp"
#include "mathlib/units/angleunit.hpp"
#include "mathlib/units/coordinate.hpp"

namespace math {

/** Cartesian point in a two-dimensional plane. */
struct Point : MathObj<Point> {
private:
    bool is_vec = false;

public:
  fieldunit<> x;
  fieldunit<> y;

  Point(fieldunit<> x_value = fieldunit<>(), fieldunit<> y_value = fieldunit<>());

  Point operator+(const Point& other) const;
  Point operator+(const fieldunit<>& value) const;
  Point operator-(const Point& other) const;
  Point operator-(const fieldunit<>& value) const;
  bool operator==(const Point& other) const;
  bool operator!=(const Point& other) const;

  /** Euclidean distance from this point to another. */
  fieldunit<> distance_to(const Point& other) const;

  /** Rotate around origin, or around the supplied point, by angle. */
  Point rotated(angle<> radians, const Point& origin = Point()) const;

  /** Return the angle from this point to another, in the default angle unit. */
  angle<> angle_to(const Point& other) const;

  /** Return the slope to another point. */
  real slope_to(const Point& other) const;
  
};

template <>
struct conversion<Point, std::string> {
    static std::string convert(const Point& point) {
        return "(" + std::to_string(point.x.get_value()) +
            ", " + std::to_string(point.y.get_value()) + ")";
    }
};

template <>
struct conversion<Point, linalg::Vector<double>> {
    static linalg::Vector<double> convert(const Point& point) {
        return linalg::Vector<double>{point.x.get_value(),
                                      point.y.get_value()};
    }
};

}  // namespace math
