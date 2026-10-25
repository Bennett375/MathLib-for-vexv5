#pragma once

#include "mathlib/abstract/object.hpp"
#include "mathlib/geometry/point.hpp"


#include <vector>

namespace math {
/** Circle represented by its center and radius. */
class Circle : public MathObj<Circle> {
public:
    Point center;
    double radius;
    Circle(const Point& center_, const double& radius_): center(center_), radius(radius_) {}
};

template <>
struct conversion<Circle, std::string> {
    static std::string convert(const Circle& circle) {
        return "Circle at: " + circle.center.as<std::string>() + " with radius: " + std::to_string(circle.radius);
    }
};

/** Return the intersection points of two circles. */
std::vector<Point> circle_intersections(const Circle& first,
                                        const Circle& second);

}