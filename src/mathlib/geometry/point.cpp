#include "point.hpp"

#include <cmath>

namespace math {

Point::Point(double x_value, double y_value) : x(x_value), y(y_value) {}

Point Point::operator+(const Point& other) const {
  return Point(x + other.x, y + other.y);
}

Point Point::operator+(double value) const {
  return Point(x + value, y + value);
}

Point Point::operator-(const Point& other) const {
  return Point(x - other.x, y - other.y);
}

Point Point::operator-(double value) const {
  return Point(x - value, y - value);
}

bool Point::operator==(const Point& other) const {
  return x == other.x && y == other.y;
}

bool Point::operator!=(const Point& other) const {
  return !(*this == other);
}

double Point::distance_to(const Point& other) const {
  return std::hypot(x - other.x, y - other.y);
}

Point Point::rotated(double radians, const Point& origin) const {
  const double offset_x = x - origin.x;
  const double offset_y = y - origin.y;
  const double cosine = std::cos(radians);
  const double sine = std::sin(radians);
  return Point(origin.x + offset_x * cosine - offset_y * sine,
               origin.y + offset_x * sine + offset_y * cosine);
}

double Point::angle_to(const Point& other) const {
  return std::atan2(other.y - y, other.x - x);
}

double Point::slope_to(const Point& other) const {
  return (other.y - y) / (other.x - x);
}

}