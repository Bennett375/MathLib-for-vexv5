#pragma once

#include "matrix.hpp"

#include <cstddef>
#include <initializer_list>

namespace math {
namespace linalg {

template <typename T>
class Vector : public Matrix<T> {
public:
    typedef Matrix<T> Base;

    explicit Vector(std::size_t size);
    Vector(std::size_t size, const T& initial_value);
    Vector(std::initializer_list<T> values);

    std::size_t size() const;
    bool empty() const;

    T& operator()(std::size_t row);
    const T& operator()(std::size_t row) const;


    Vector operator+(const Vector& other) const;
    Vector operator-(const Vector& other) const;
    Vector operator-() const;

    Vector operator*(const T& scalar) const;
    Vector operator/(const T& scalar) const;

    Vector& operator+=(const Vector& other);
    Vector& operator-=(const Vector& other);
    Vector& operator*=(const T& scalar);

    double norm() const;
    double squared_norm() const;
    double distance_to(const Vector& other) const;

    Vector<double> normalized() const;

    double angle_between(const Vector& other) const;

    Vector<double> projection_onto(const Vector& other) const;

    Matrix<T> mat_from_diag() const;

    static Vector zeros(std::size_t size);
};

template <typename T>
double dot(const Vector<T>& vec1, const Vector<T>& vec2);

template <typename T>
Vector<T> cross(const Vector<T>& vec1, const Vector<T>& vec2);

template <typename T>
Vector<T> operator*(const T& scalar, const Vector<T>& vector);

extern template class Vector<int>;
extern template class Vector<float>;
extern template class Vector<double>;

extern template double dot<int>(
    const Vector<int>&, const Vector<int>&);
extern template double dot<float>(
    const Vector<float>&, const Vector<float>&);
extern template double dot<double>(
    const Vector<double>&, const Vector<double>&);

extern template Vector<int> cross<int>(
    const Vector<int>&, const Vector<int>&);
extern template Vector<float> cross<float>(
    const Vector<float>&, const Vector<float>&);
extern template Vector<double> cross<double>(
    const Vector<double>&, const Vector<double>&);

extern template Vector<int> operator*(
    const int&, const Vector<int>&);
extern template Vector<float> operator*(
    const float&, const Vector<float>&);
extern template Vector<double> operator*(
    const double&, const Vector<double>&);

typedef Vector<double> vec;

} // namespace linalg

template <>
struct conversion<linalg::Vector<double>, std::string> {
    static std::string convert(const linalg::Vector<double>& vec) {
        return conversion<linalg::Matrix<double>, std::string>::convert(vec);
    }
};
} // namespace math