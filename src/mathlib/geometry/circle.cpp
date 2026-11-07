#include "mathlib/geometry/circle.hpp"
#include "mathlib/arithmetic/arithmetic.hpp"

#include <cmath>

namespace math {
std::vector<Point> circle_intersections(const Circle& first,
                                       const Circle& second) {
  const double dx = second.center.x.get_value() - first.center.x.get_value();
  const double dy = second.center.y.get_value() - first.center.y.get_value();
  const double distance = std::hypot(dx, dy);
  const double first_radius = first.radius.get_value();
  const double second_radius = second.radius.get_value();

  if (distance > first_radius + second_radius ||
      distance < std::fabs(first_radius - second_radius) ||
      distance == 0.0) {
    return std::vector<Point>();
  }

  const double along =
      (sqr(first_radius).get_value() - sqr(second_radius).get_value() +
       sqr(distance).get_value()) /
      (2.0 * distance);
  const double height_squared =
      sqr(first_radius).get_value() - sqr(along).get_value();
  const double height = sqrt(std::max(0.0, height_squared)).get_value();
  const double midpoint_x = first.center.x.get_value() + along * dx / distance;
  const double midpoint_y = first.center.y.get_value() + along * dy / distance;
  const Point one(midpoint_x + height * dy / distance,
                  midpoint_y - height * dx / distance);

  if (height == 0.0) {
    return std::vector<Point>(1, one);
  }

  const Point two(midpoint_x - height * dy / distance,
                  midpoint_y + height * dx / distance);
  std::vector<Point> result;
  result.push_back(one);
  result.push_back(two);
  return result;
}
}