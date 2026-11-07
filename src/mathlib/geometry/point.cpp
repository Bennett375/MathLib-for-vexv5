#include "mathlib/geometry/point.hpp"

#include <cmath>

namespace math {

Point::Point(fieldunit<> x_value, fieldunit<> y_value) : x(x_value), y(y_value) {}

Point Point::operator+(const Point& other) const {
  return Point(x + other.x, y + other.y);
}

Point Point::operator+(const fieldunit<>& value) const {
  return Point(x + value, y + value);
}

Point Point::operator-(const Point& other) const {
  return Point(x - other.x, y - other.y);
}

Point Point::operator-(const fieldunit<>& value) const {
  return Point(x - value, y - value);
}

bool Point::operator==(const Point& other) const {
  return x == other.x && y == other.y;
}

bool Point::operator!=(const Point& other) const {
  return !(*this == other);
}

fieldunit<> Point::distance_to(const Point& other) const {
  return fieldunit<>(std::hypot(x.get_value() - other.x.get_value(),
                                y.get_value() - other.y.get_value()));
}

Point Point::rotated(angle<> angle_value, const Point& origin) const {
  const double radians = angle_value.get_value();
  const double offset_x = x.get_value() - origin.x.get_value();
  const double offset_y = y.get_value() - origin.y.get_value();
  const double cosine = std::cos(radians);
  const double sine = std::sin(radians);
  return Point(origin.x.get_value() + offset_x * cosine - offset_y * sine,
               origin.y.get_value() + offset_x * sine + offset_y * cosine);
}

angle<> Point::angle_to(const Point& other) const {
  return angle<>(std::atan2(other.y.get_value() - y.get_value(),
                            other.x.get_value() - x.get_value()));
}

real Point::slope_to(const Point& other) const {
  return real((other.y.get_value() - y.get_value()) /
              (other.x.get_value() - x.get_value()));
}

}
