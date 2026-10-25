#pragma once

#include <vector>

namespace math {

/** Constant pi. */
static const double PI = 3.14159265358979323846;
/** Constant 2*pi. */
static const double TAU = 2.0 * PI;

static const double e = 2.718281828459045;

/** Return the arithmetic mean of the values, or zero for an empty vector. */
double mean(const std::vector<double>& values);
/** Return the mean of two values. */
double mean(double first, double second);

/** Return -1, 0, or 1 according to the sign of the value. */
int sign(int value);
/** Return -1, 0, or 1 according to the sign of the value. */
double sign(double value);

/**
 * Partition values into groups of at most group_size.
 * @param values Source sequence.
 * @param group_size Maximum number of values in each group; zero returns none.
 * @param disjoint If true, groups do not overlap; otherwise step by one.
 */
template <typename T>
std::vector<std::vector<T> > make_groups(
    const std::vector<T>& values, std::size_t group_size,
    bool disjoint = true) {
  std::vector<std::vector<T> > result;
  if (group_size == 0) {
    return result;
  }

  const std::size_t step = disjoint ? group_size : 1;
  for (std::size_t begin = 0; begin < values.size(); begin += step) {
    const std::size_t end =
        begin + group_size < values.size() ? begin + group_size : values.size();
    result.push_back(
        std::vector<T>(values.begin() + begin, values.begin() + end));
  }
  return result;
}

/** Partition values into pairs, optionally overlapping by one element. */
template <typename T>
std::vector<std::vector<T> > make_pairs(
    const std::vector<T>& values, bool disjoint = true) {
  return make_groups(values, 2, disjoint);
}

/** Clamp a value to the inclusive [minimum, maximum] interval. */
double clamp(double value, double minimum = -100.0, double maximum = 100.0);

/** Raise base to a real exponent. */
double power(double base, double exponent);

/** Natural exponential */
double exp(double exponent);

/** Return the square of value. */
double sqr(double value);
/** Return the square root of value. */
double sqrt(double value);

}