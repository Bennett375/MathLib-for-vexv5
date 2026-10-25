#pragma once

#include "mathlib/abstract/object.hpp"

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace math {
namespace linalg {

template <typename T>
class Vector;

template <typename T>
class Matrix : public MathObj<Matrix<T>> {
public:
    // Constructors
    Matrix();
    Matrix(std::size_t rows, std::size_t cols);
    Matrix(std::size_t rows, std::size_t cols, const T& initial_value);
    Matrix(std::initializer_list<std::initializer_list<T>> values);

    // Dimensions
    std::size_t rows() const;
    std::size_t cols() const;
    bool empty() const;
    bool is_square() const;

    // Element access
    T& operator()(std::size_t row, std::size_t col);
    const T& operator()(std::size_t row, std::size_t col) const;

    T& at(std::size_t row, std::size_t col);
    const T& at(std::size_t row, std::size_t col) const;

    // Data access
    const std::vector<T>& data() const;
    void fill(const T& value);

    // Matrix arithmetic
    Matrix<T> operator+(const Matrix<T>& other) const;
    Matrix<T> operator-(const Matrix<T>& other) const;
    Matrix<T> operator*(const Matrix<T>& other) const;
    Vector<T> operator*(const Vector<T>& other) const;

    Matrix<T> operator*(const T& scalar) const;
    Matrix<T> operator/(const T& scalar) const;

    Matrix<T>& operator+=(const Matrix<T>& other);
    Matrix<T>& operator-=(const Matrix<T>& other);
    Matrix<T>& operator*=(const T& scalar);

    // Linear algebra
    Matrix<T> transpose() const;
    T determinant() const;
    Matrix<T> inverse() const;

    // Matrix factories
    static Matrix<T> identity(std::size_t size);
    static Matrix<T> zeros(std::size_t rows, std::size_t cols);

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<T> data_;

    std::size_t index(std::size_t row, std::size_t col) const;
    void require_same_dimensions(const Matrix<T>& other) const;
};

// Scalar multiplication with the scalar on the left.
template <typename T>
Matrix<T> operator*(const T& scalar, const Matrix<T>& matrix);

extern template class Matrix<int>;
extern template class Matrix<float>;
extern template class Matrix<double>;

extern template Matrix<int> operator*(
    const int&, const Matrix<int>&);

extern template Matrix<float> operator*(
    const float&, const Matrix<float>&);

extern template Matrix<double> operator*(
    const double&, const Matrix<double>&);

} // namespace linalg

template <>
struct conversion<linalg::Matrix<double>, std::string> {
    static std::string convert(const linalg::Matrix<double>& mat) {
        std::string string;
        for (int i = 0; i < mat.rows(); ++i) {
            string.append("|");
            for (int j = 0; j < mat.cols(); ++j) {
                string.append(std::to_string(mat(i, j)));
            }
            string.append("|\n");
        }
        return string;
    }
};

} // namespace math

