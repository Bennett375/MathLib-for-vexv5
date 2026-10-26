#include "mathlib/linalg/matrix.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace math {
namespace linalg {

template <typename T>
Matrix<T>::Matrix()
    : rows_(0), cols_(0), data_() {}

template <typename T>
Matrix<T>::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, T()) {}

template <typename T>
Matrix<T>::Matrix(
    std::size_t rows,
    std::size_t cols,
    const T& initial_value)
    : rows_(rows),
      cols_(cols),
      data_(rows * cols, initial_value) {}

template <typename T>
Matrix<T>::Matrix(
    std::initializer_list<std::initializer_list<T>> values)
    : rows_(values.size()), cols_(0), data_() {

    if (values.size() == 0) {
        return;
    }

    cols_ = values.begin()->size();

    for (const auto& row : values) {
        if (row.size() != cols_) {
            throw std::invalid_argument(
                "Matrix initializer rows must have equal lengths.");
        }

        data_.insert(data_.end(), row.begin(), row.end());
    }
}

template <typename T>
std::size_t Matrix<T>::rows() const {
    return rows_;
}

template <typename T>
std::size_t Matrix<T>::cols() const {
    return cols_;
}

template <typename T>
bool Matrix<T>::empty() const {
    return rows_ == 0 || cols_ == 0;
}

template <typename T>
bool Matrix<T>::is_square() const {
    return rows_ == cols_;
}

// Element access

template <typename T>
std::size_t Matrix<T>::index(
    std::size_t row,
    std::size_t col) const {

    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range.");
    }

    return row * cols_ + col;
}

template <typename T>
T& Matrix<T>::operator()(
    std::size_t row,
    std::size_t col) {

    return data_[index(row, col)];
}

template <typename T>
const T& Matrix<T>::operator()(
    std::size_t row,
    std::size_t col) const {

    return data_[index(row, col)];
}

template <typename T>
T& Matrix<T>::at(
    std::size_t row,
    std::size_t col) {

    return data_[index(row, col)];
}

template <typename T>
const T& Matrix<T>::at(
    std::size_t row,
    std::size_t col) const {

    return data_[index(row, col)];
}

// Data access

template <typename T>
const std::vector<T>& Matrix<T>::data() const {
    return data_;
}

template <typename T>
void Matrix<T>::fill(const T& value) {
    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] = value;
    }
}

// Validation

template <typename T>
void Matrix<T>::require_same_dimensions(
    const Matrix<T>& other) const {

    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(
            "Matrix dimensions do not match.");
    }
}

// Matrix addition and subtraction

template <typename T>
Matrix<T> Matrix<T>::operator+(
    const Matrix<T>& other) const {

    require_same_dimensions(other);

    Matrix<T> result(rows_, cols_);

    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(
    const Matrix<T>& other) const {

    require_same_dimensions(other);

    Matrix<T> result(rows_, cols_);

    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }

    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(
    const Matrix<T>& other) {

    require_same_dimensions(other);

    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }

    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(
    const Matrix<T>& other) {

    require_same_dimensions(other);

    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(
    const Matrix<T>& other) const {

    if (cols_ != other.rows_) {
        throw std::invalid_argument(
            "Matrix multiplication requires left.cols == right.rows.");
    }

    Matrix<T> result(rows_, other.cols_, T());

    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t k = 0; k < cols_; ++k) {
            const T left = (*this)(i, k);

            for (std::size_t j = 0; j < other.cols_; ++j) {
                result(i, j) += left * other(k, j);
            }
        }
    }

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const T& scalar) const {
    Matrix<T> result(rows_, cols_);

    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator/(const T& scalar) const {
    if (scalar == T()) {
        throw std::invalid_argument(
            "Cannot divide a matrix by zero.");
    }

    Matrix<T> result(rows_, cols_);

    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] / scalar;
    }

    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*=(const T& scalar) {
    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] *= scalar;
    }

    return *this;
}

template <typename T>
Matrix<T> operator*(
    const T& scalar,
    const Matrix<T>& matrix) {

    return matrix * scalar;
}

template <typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result(cols_, rows_);

    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}

template <typename T>
T Matrix<T>::determinant() const {
    if (!is_square()) {
        throw std::invalid_argument(
            "Determinant requires a square matrix.");
    }

    if (rows_ == 0) {
        return T(1);
    }

    const std::size_t n = rows_;
    std::vector<long double> work;
    work.reserve(data_.size());

    for (std::size_t i = 0; i < data_.size(); ++i) {
        work.push_back(static_cast<long double>(data_[i]));
    }

    long double det = 1.0L;
    int sign = 1;

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot_row = col;
        long double pivot_abs =
            std::fabs(work[col * n + col]);

        for (std::size_t row = col + 1; row < n; ++row) {
            const long double candidate =
                std::fabs(work[row * n + col]);

            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }

        if (pivot_abs == 0.0L) {
            return T();
        }

        if (pivot_row != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(
                    work[col * n + j],
                    work[pivot_row * n + j]);
            }

            sign = -sign;
        }

        const long double pivot = work[col * n + col];
        det *= pivot;

        for (std::size_t row = col + 1; row < n; ++row) {
            const long double factor =
                work[row * n + col] / pivot;

            for (std::size_t j = col + 1; j < n; ++j) {
                work[row * n + j] -=
                    factor * work[col * n + j];
            }
        }
    }

    det *= static_cast<long double>(sign);
    return static_cast<T>(det);
}

template <typename T>
Matrix<T> Matrix<T>::inverse() const {
    if (!is_square()) {
        throw std::invalid_argument(
            "Inverse requires a square matrix.");
    }

    const std::size_t n = rows_;

    if (n == 0) {
        return Matrix<T>();
    }

    // Work in long double to reduce intermediate rounding.
    // The final result is converted back to T.
    std::vector<long double> left(n * n);
    std::vector<long double> right(n * n, 0.0L);

    for (std::size_t i = 0; i < n * n; ++i) {
        left[i] = static_cast<long double>(data_[i]);
    }

    for (std::size_t i = 0; i < n; ++i) {
        right[i * n + i] = 1.0L;
    }

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot_row = col;
        long double pivot_abs =
            std::fabs(left[col * n + col]);

        for (std::size_t row = col + 1; row < n; ++row) {
            const long double candidate =
                std::fabs(left[row * n + col]);

            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }

        if (pivot_abs <=
            std::numeric_limits<long double>::epsilon()) {
            throw std::domain_error(
                "Cannot invert a singular or numerically singular matrix.");
        }

        if (pivot_row != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(
                    left[col * n + j],
                    left[pivot_row * n + j]);

                std::swap(
                    right[col * n + j],
                    right[pivot_row * n + j]);
            }
        }

        const long double pivot = left[col * n + col];

        for (std::size_t j = 0; j < n; ++j) {
            left[col * n + j] /= pivot;
            right[col * n + j] /= pivot;
        }

        for (std::size_t row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }

            const long double factor = left[row * n + col];

            for (std::size_t j = 0; j < n; ++j) {
                left[row * n + j] -=
                    factor * left[col * n + j];

                right[row * n + j] -=
                    factor * right[col * n + j];
            }
        }
    }

    Matrix<T> result(n, n);

    for (std::size_t i = 0; i < n * n; ++i) {
        result.data_[i] = static_cast<T>(right[i]);
    }

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::identity(std::size_t size) {
    Matrix<T> result(size, size, T());

    for (std::size_t i = 0; i < size; ++i) {
        result(i, i) = T(1);
    }

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::zeros(
    std::size_t rows,
    std::size_t cols) {

    return Matrix<T>(rows, cols, T());
}

template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;

template Matrix<int> operator*(
    const int&, const Matrix<int>&);

template Matrix<float> operator*(
    const float&, const Matrix<float>&);

template Matrix<double> operator*(
    const double&, const Matrix<double>&);
    
} // namespace linalg
} // namespace math