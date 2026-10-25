#pragma once

#include "mathlib/abstract/object.hpp"
#include "mathlib/linalg/vector.hpp"

namespace math {

/** Cartesian point in a two-dimensional plane. */
struct Point : MathObj<Point> {
private:
    bool is_vec = false;

public:
  double x;
  double y;

  Point(double x_value = 0.0, double y_value = 0.0);

  Point operator+(const Point& other) const;
  Point operator+(double value) const;
  Point operator-(const Point& other) const;
  Point operator-(double value) const;
  bool operator==(const Point& other) const;
  bool operator!=(const Point& other) const;

  /** Euclidean distance from this point to another. */
  double distance_to(const Point& other) const;

  /** Rotate around origin, or around the supplied point, by radians. */
  Point rotated(double radians, const Point& origin = Point()) const;

  /** Return the angle from this point to another, in radians. */
  double angle_to(const Point& other) const;

  /** Return the slope to another point. */
  double slope_to(const Point& other) const;
  
};

template <>
struct conversion<Point, std::string> {
    static std::string convert(const Point& point) {
        return "(" + std::to_string(point.x) +
            ", " + std::to_string(point.y) + ")";
    }
};

template <>
struct conversion<Point, linalg::Vector<double>> {
    static linalg::Vector<double> convert(const Point& point) {
        return linalg::Vector<double>{point.x, point.y};
    }
};

}  // namespace math
