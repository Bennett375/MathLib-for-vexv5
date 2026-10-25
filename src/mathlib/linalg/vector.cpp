#include "vector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace math {
namespace linalg {

// -----------------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------------

template <typename T>
Vector<T>::Vector(std::size_t size)
    : Base(size, 1) {}

template <typename T>
Vector<T>::Vector(std::size_t size, const T& initial_value)
    : Base(size, 1, initial_value) {}

template <typename T>
Vector<T>::Vector(std::initializer_list<T> values)
    : Base(values.size(), 1) {

    std::size_t i = 0;

    for (const T& value : values) {
        (*this)(i++) = value;
    }
}

// -----------------------------------------------------------------------------
// Size
// -----------------------------------------------------------------------------

template <typename T>
std::size_t Vector<T>::size() const {
    return this->rows();
}

template <typename T>
bool Vector<T>::empty() const {
    return this->rows() == 0;
}

template <typename T>
T& Vector<T>::operator()(std::size_t row) {
    if (row >= size()) {
        throw std::out_of_range("Vector index out of range");
    }
    return this->Matrix<T>::at(row, 0);
}

template <typename T>
const T& Vector<T>::operator()(std::size_t row) const {
    if (row >= size()) {
        throw std::out_of_range("Vector index out of range");
    }
    return this->Matrix<T>::at(row, 0);
}

// -----------------------------------------------------------------------------
// Validation helper
// -----------------------------------------------------------------------------

template <typename T>
static void require_same_size(
    const Vector<T>& a,
    const Vector<T>& b) {

    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vector dimensions do not match.");
    }
}

// -----------------------------------------------------------------------------
// Arithmetic
// -----------------------------------------------------------------------------

template <typename T>
Vector<T> Vector<T>::operator+(const Vector<T>& other) const {
    require_same_size(*this, other);

    Vector<T> result(this->size());

    for (std::size_t i = 0; i < this->size(); ++i) {
        result(i) = (*this)(i) + other(i);
    }

    return result;
}

template <typename T>
Vector<T> Vector<T>::operator-(const Vector<T>& other) const {
    require_same_size(*this, other);

    Vector<T> result(this->size());

    for (std::size_t i = 0; i < this->size(); ++i) {
        result(i) = (*this)(i) - other(i);
    }

    return result;
}

template <typename T>
Vector<T> Vector<T>::operator-() const {
    Vector<T> result(this->size());

    for (std::size_t i = 0; i < this->size(); ++i) {
        result(i) = -(*this)(i);
    }

    return result;
}

template <typename T>
Vector<T> Vector<T>::operator*(const T& scalar) const {
    Vector<T> result(this->size());

    for (std::size_t i = 0; i < this->size(); ++i) {
        result(i) = (*this)(i) * scalar;
    }

    return result;
}

template <typename T>
Vector<T> Vector<T>::operator/(const T& scalar) const {
    if (scalar == T()) {
        throw std::invalid_argument(
            "Cannot divide a vector by zero.");
    }

    Vector<T> result(this->size());

    for (std::size_t i = 0; i < this->size(); ++i) {
        result(i) = (*this)(i) / scalar;
    }

    return result;
}

template <typename T>
Vector<T>& Vector<T>::operator+=(const Vector<T>& other) {
    require_same_size(*this, other);

    for (std::size_t i = 0; i < this->size(); ++i) {
        (*this)(i) += other(i);
    }

    return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator-=(const Vector<T>& other) {
    require_same_size(*this, other);

    for (std::size_t i = 0; i < this->size(); ++i) {
        (*this)(i) -= other(i);
    }

    return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator*=(const T& scalar) {
    for (std::size_t i = 0; i < this->size(); ++i) {
        (*this)(i) *= scalar;
    }

    return *this;
}

template <typename T>
Vector<T> operator*(const T& scalar, const Vector<T>& vector) {
    return vector * scalar;
}

// -----------------------------------------------------------------------------
// Matrix-vector product
// -----------------------------------------------------------------------------

template <typename T>
Vector<T> Matrix<T>::operator*(const Vector<T>& other) const {
    if (this->cols() != other.size()) {
        throw std::invalid_argument(
            "Matrix-vector multiplication requires matrix.cols == vector.size.");
    }

    Vector<T> result(this->rows());

    for (std::size_t i = 0; i < this->rows(); ++i) {
        T sum = T();

        for (std::size_t j = 0; j < this->cols(); ++j) {
            sum += (*this)(i, j) * other(j);
        }

        result(i) = sum;
    }

    return result;
}

// -----------------------------------------------------------------------------
// Dot product
// Accumulates in double to reduce intermediate precision loss.
// -----------------------------------------------------------------------------

template <typename T>
double dot(const Vector<T>& vec1, const Vector<T>& vec2) {
    require_same_size(vec1, vec2);

    double result = 0.0;

    for (std::size_t i = 0; i < vec1.size(); ++i) {
        result += static_cast<double>(vec1(i)) *
                  static_cast<double>(vec2(i));
    }

    return result;
}

// -----------------------------------------------------------------------------
// Cross product
// Defined for three-dimensional vectors.
// -----------------------------------------------------------------------------

template <typename T>
Vector<T> cross(const Vector<T>& vec1, const Vector<T>& vec2) {
    if (vec1.size() != 3 || vec2.size() != 3) {
        throw std::invalid_argument(
            "Cross product requires two 3D vectors.");
    }

    return Vector<T>{
        static_cast<T>(
            vec1(1) * vec2(2) -
            vec1(2) * vec2(1)),

        static_cast<T>(
            vec1(2) * vec2(0) -
            vec1(0) * vec2(2)),

        static_cast<T>(
            vec1(0) * vec2(1) -
            vec1(1) * vec2(0))
    };
}

// -----------------------------------------------------------------------------
// Norms
// -----------------------------------------------------------------------------

template <typename T>
double Vector<T>::squared_norm() const {
    return dot(*this, *this);
}

template <typename T>
double Vector<T>::norm() const {
    return std::sqrt(squared_norm());
}

// -----------------------------------------------------------------------------
// Normalization
// Returns a double-precision vector to preserve fractional components.
// -----------------------------------------------------------------------------

template <typename T>
Vector<double> Vector<T>::normalized() const {
    const double magnitude = norm();

    if (magnitude == 0.0) {
        throw std::domain_error(
            "Cannot normalize a zero vector.");
    }

    Vector<double> result(this->size());

    for (std::size_t i = 0; i < this->size(); ++i) {
        result(i) =
            static_cast<double>((*this)(i)) / magnitude;
    }

    return result;
}

// -----------------------------------------------------------------------------
// Distance
// -----------------------------------------------------------------------------

template <typename T>
double Vector<T>::distance_to(const Vector<T>& other) const {
    require_same_size(*this, other);

    double sum = 0.0;

    for (std::size_t i = 0; i < this->size(); ++i) {
        const double difference =
            static_cast<double>((*this)(i)) -
            static_cast<double>(other(i));

        sum += difference * difference;
    }

    return std::sqrt(sum);
}

// -----------------------------------------------------------------------------
// Angle between vectors
// Returns radians in [0, pi].
// -----------------------------------------------------------------------------

template <typename T>
double Vector<T>::angle_between(const Vector<T>& other) const {
    require_same_size(*this, other);

    const double norm_a = norm();
    const double norm_b = other.norm();

    if (norm_a == 0.0 || norm_b == 0.0) {
        throw std::domain_error(
            "Angle is undefined for a zero vector.");
    }

    double cosine = dot(*this, other) / (norm_a * norm_b);

    // Protect acos against floating-point rounding errors.
    cosine = std::max(-1.0, std::min(1.0, cosine));

    return std::acos(cosine);
}

// -----------------------------------------------------------------------------
// Projection
// Projects this vector onto 'other'.
// -----------------------------------------------------------------------------

template <typename T>
Vector<double> Vector<T>::projection_onto(
    const Vector<T>& other) const {

    require_same_size(*this, other);

    const double denominator = dot(other, other);

    if (denominator == 0.0) {
        throw std::domain_error(
            "Cannot project onto a zero vector.");
    }

    const double scale = dot(*this, other) / denominator;

    Vector<double> result(this->size());

    for (std::size_t i = 0; i < this->size(); ++i) {
        result(i) =
            scale * static_cast<double>(other(i));
    }

    return result;
}

// -----------------------------------------------------------------------------
// Diagonal matrix
// -----------------------------------------------------------------------------

template <typename T>
Matrix<T> Vector<T>::mat_from_diag() const {
    linalg::Matrix<T> result(
        this->size(), this->size(), T());

    for (std::size_t i = 0; i < this->size(); ++i) {
        result(i, i) = (*this)(i);
    }

    return result;
}

// -----------------------------------------------------------------------------
// Factory
// -----------------------------------------------------------------------------

template <typename T>
Vector<T> Vector<T>::zeros(std::size_t size) {
    return Vector<T>(size, T());
}

// -----------------------------------------------------------------------------
// Explicit instantiations
// -----------------------------------------------------------------------------

template class Vector<int>;
template class Vector<float>;
template class Vector<double>;

template double dot<int>(
    const Vector<int>&, const Vector<int>&);
template double dot<float>(
    const Vector<float>&, const Vector<float>&);
template double dot<double>(
    const Vector<double>&, const Vector<double>&);

template Vector<int> cross<int>(
    const Vector<int>&, const Vector<int>&);
template Vector<float> cross<float>(
    const Vector<float>&, const Vector<float>&);
template Vector<double> cross<double>(
    const Vector<double>&, const Vector<double>&);

template Vector<int> operator*(
    const int&, const Vector<int>&);
template Vector<float> operator*(
    const float&, const Vector<float>&);
template Vector<double> operator*(
    const double&, const Vector<double>&);

template Vector<int> Matrix<int>::operator*(
    const Vector<int>&) const;
template Vector<float> Matrix<float>::operator*(
    const Vector<float>&) const;
template Vector<double> Matrix<double>::operator*(
    const Vector<double>&) const;

} // namespace linalg
} // namespace math