#pragma once

#include <vector>

#include "mathlib/linalg/vector.hpp"
#include "mathlib/linalg/matrix.hpp"
#include "mathlib/units/numbers.hpp"

namespace math {

/** Return the arithmetic mean of the values, or zero for an empty vector. */
real mean(const std::vector<double>& values);
/** Return the mean of two values. */
real mean(real first, real second);
/** Return the average of a matrix, or zero for an empty matrix. */
inline real mean(const math::linalg::Matrix<double>& mat) {
  return mat.empty()
      ? real(0.0)
      : real(mat.sum() / static_cast<double>(mat.size()));
}
/** Return the average of a vector, or zero for an empty vector. */
inline real mean(const math::linalg::Vector<double>& vec) {
  return vec.empty()
      ? real(0.0)
      : real(vec.sum() / static_cast<double>(vec.size()));
}

/** Return -1, 0, or 1 according to the sign of the value. */
int sign(int value);
/** Return -1, 0, or 1 according to the sign of the value. */
real sign(double value);
/** Return -1, 0, or 1 according to the sign of the value. */
real sign(real value);

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
  std::vector<std::vector<T> > result;
  if (values.size() < 2) {
    return result;
  }

  const std::size_t step = disjoint ? 2 : 1;
  for (std::size_t begin = 0; begin + 1 < values.size(); begin += step) {
    result.push_back(
        std::vector<T>(values.begin() + begin, values.begin() + begin + 2));
  }
  return result;
}

/** Clamp a value to the inclusive [minimum, maximum] interval. */
real clamp(real value, real minimum = -100.0, real maximum = 100.0);

/** Raise base to a real exponent. */
real power(real base, real exponent);

/** Natural exponential */
real exp(real exponent);

/** Return the square of value. */
real sqr(real value);
/** Return the square root of value. */
real sqrt(real value);

}
