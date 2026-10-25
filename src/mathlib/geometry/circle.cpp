#include "circle.hpp"
#include "mathlib/arithmetic/arithmetic.hpp"

#include <cmath>

namespace math {
std::vector<Point> circle_intersections(const Circle& first,
                                       const Circle& second) {
  const double dx = second.center.x - first.center.x;
  const double dy = second.center.y - first.center.y;
  const double distance = std::hypot(dx, dy);

  if (distance > first.radius + second.radius ||
      distance < std::fabs(first.radius - second.radius) ||
      distance == 0.0) {
    return std::vector<Point>();
  }

  const double along =
      (sqr(first.radius) - sqr(second.radius) + sqr(distance)) /
      (2.0 * distance);
  const double height_squared = sqr(first.radius) - sqr(along);
  const double height = sqrt(std::max(0.0, height_squared));
  const double midpoint_x = first.center.x + along * dx / distance;
  const double midpoint_y = first.center.y + along * dy / distance;
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