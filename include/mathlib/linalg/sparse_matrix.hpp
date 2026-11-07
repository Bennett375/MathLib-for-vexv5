#pragma once

#include "mathlib/abstract/object.hpp"
#include "mathlib/linalg/matrix.hpp"
#include "mathlib/linalg/matrix_storage.hpp"
#include "mathlib/linalg/sparse_vector.hpp"
#include "mathlib/linalg/vector.hpp"

#include <cstddef>
#include <map>
#include <stdexcept>
#include <utility>

namespace math {
namespace linalg {

template <typename T>
class SparseMatrix : public MathObj<SparseMatrix<T>>,
                     public MatrixStorage<
                         T, detail::SparseMatrixStorage<T>> {
public:
    typedef MatrixStorage<T, detail::SparseMatrixStorage<T>> Base;
    typedef std::pair<std::size_t, std::size_t> Index;
    typedef detail::SparseMatrixStorage<T> Storage;

    SparseMatrix() : Base() {}

    SparseMatrix(std::size_t rows, std::size_t cols)
        : Base(rows, cols) {}

    explicit SparseMatrix(const Matrix<T>& matrix)
        : Base(matrix.rows(), matrix.cols()) {
        for (std::size_t row = 0; row < this->rows_; ++row) {
            for (std::size_t col = 0; col < this->cols_; ++col) {
                set(row, col, matrix(row, col));
            }
        }
    }

    std::size_t rows() const { return this->rows_; }
    std::size_t cols() const { return this->cols_; }
    std::size_t size() const { return this->rows_ * this->cols_; }
    std::size_t non_zero_count() const { return this->data_.size(); }
    bool empty() const { return this->rows_ == 0 || this->cols_ == 0; }
    bool is_square() const { return this->rows_ == this->cols_; }

    T operator()(std::size_t row, std::size_t col) const {
        require_index(row, col);
        return this->storage_value(row, col);
    }

    void set(std::size_t row, std::size_t col, const T& value) {
        require_index(row, col);
        this->set_storage_value(row, col, value);
    }

    void erase(std::size_t row, std::size_t col) {
        require_index(row, col);
        this->data_.erase(Index(row, col));
    }

    void clear() { this->data_.clear(); }

    void fill(const T& value) {
        this->fill_storage(value);
    }

    const Storage& data() const { return this->data_; }

    double sum() const {
        return this->storage_sum();
    }

    Matrix<T> to_dense() const {
        Matrix<T> result(this->rows_, this->cols_);
        for (typename Storage::const_iterator it = this->data_.begin();
             it != this->data_.end(); ++it) {
            result(it->first.first, it->first.second) = it->second;
        }
        return result;
    }

    SparseMatrix transpose() const {
        SparseMatrix result(this->cols_, this->rows_);
        for (typename Storage::const_iterator it = this->data_.begin();
             it != this->data_.end(); ++it) {
            result.set(it->first.second, it->first.first, it->second);
        }
        return result;
    }

    SparseMatrix operator+(const SparseMatrix& other) const {
        require_same_dimensions(other);
        SparseMatrix result(*this);
        for (typename Storage::const_iterator it = other.data_.begin();
             it != other.data_.end(); ++it) {
            result.set(it->first.first, it->first.second,
                       result(it->first.first, it->first.second) + it->second);
        }
        return result;
    }

    SparseMatrix operator-(const SparseMatrix& other) const {
        require_same_dimensions(other);
        SparseMatrix result(*this);
        for (typename Storage::const_iterator it = other.data_.begin();
             it != other.data_.end(); ++it) {
            result.set(it->first.first, it->first.second,
                       result(it->first.first, it->first.second) - it->second);
        }
        return result;
    }

    SparseMatrix operator*(const T& scalar) const {
        SparseMatrix result(this->rows_, this->cols_);
        if (scalar == T()) {
            return result;
        }
        for (typename Storage::const_iterator it = this->data_.begin();
             it != this->data_.end(); ++it) {
            result.set(it->first.first, it->first.second, it->second * scalar);
        }
        return result;
    }

    SparseMatrix operator/(const T& scalar) const {
        if (scalar == T()) {
            throw std::invalid_argument(
                "Cannot divide a sparse matrix by zero.");
        }
        SparseMatrix result(this->rows_, this->cols_);
        for (typename Storage::const_iterator it = this->data_.begin();
             it != this->data_.end(); ++it) {
            result.set(it->first.first, it->first.second, it->second / scalar);
        }
        return result;
    }

    SparseMatrix operator*(const SparseMatrix& other) const {
        if (this->cols_ != other.rows_) {
            throw std::invalid_argument(
                "Sparse matrix multiplication requires left.cols == right.rows.");
        }

        SparseMatrix result(this->rows_, other.cols_);
        Storage accumulated;
        for (typename Storage::const_iterator left = this->data_.begin();
             left != this->data_.end(); ++left) {
            const std::size_t row = left->first.first;
            const std::size_t inner = left->first.second;
            typename Storage::const_iterator right =
                other.data_.lower_bound(Index(inner, 0));
            while (right != other.data_.end() &&
                   right->first.first == inner) {
                accumulated[Index(row, right->first.second)] +=
                    left->second * right->second;
                ++right;
            }
        }
        for (typename Storage::const_iterator it = accumulated.begin();
             it != accumulated.end(); ++it) {
            result.set(it->first.first, it->first.second, it->second);
        }
        return result;
    }

    SparseVector<T> operator*(const SparseVector<T>& vector) const {
        if (this->cols_ != vector.size()) {
            throw std::invalid_argument(
                "Sparse matrix-vector multiplication requires matrix.cols == vector.size.");
        }
        SparseVector<T> result(this->rows_);
        for (typename Storage::const_iterator it = this->data_.begin();
             it != this->data_.end(); ++it) {
            const T value = vector(it->first.second);
            if (value != T()) {
                const std::size_t row = it->first.first;
                result.set(row, result(row) + it->second * value);
            }
        }
        return result;
    }

    Vector<T> operator*(const Vector<T>& vector) const {
        if (this->cols_ != vector.size()) {
            throw std::invalid_argument(
                "Sparse matrix-vector multiplication requires matrix.cols == vector.size.");
        }
        Vector<T> result(this->rows_);
        for (typename Storage::const_iterator it = this->data_.begin();
             it != this->data_.end(); ++it) {
            result(it->first.first) +=
                it->second * vector(it->first.second);
        }
        return result;
    }

    SparseMatrix& operator+=(const SparseMatrix& other) {
        *this = *this + other;
        return *this;
    }

    SparseMatrix& operator-=(const SparseMatrix& other) {
        *this = *this - other;
        return *this;
    }

    SparseMatrix& operator*=(const T& scalar) {
        if (scalar == T()) {
            clear();
            return *this;
        }
        typename Storage::iterator it = this->data_.begin();
        while (it != this->data_.end()) {
            it->second *= scalar;
            if (it->second == T()) {
                it = this->data_.erase(it);
            } else {
                ++it;
            }
        }
        return *this;
    }

    static SparseMatrix identity(std::size_t size) {
        SparseMatrix result(size, size);
        for (std::size_t i = 0; i < size; ++i) {
            result.set(i, i, T(1));
        }
        return result;
    }

    static SparseMatrix zeros(std::size_t rows, std::size_t cols) {
        return SparseMatrix(rows, cols);
    }

private:
    using Base::rows_;
    using Base::cols_;
    using Base::data_;

    void require_index(std::size_t row, std::size_t col) const {
        if (row >= this->rows_ || col >= this->cols_) {
            throw std::out_of_range("Sparse matrix index out of range.");
        }
    }

    void require_same_dimensions(const SparseMatrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument(
                "Sparse matrix dimensions do not match.");
        }
    }
};

template <typename T>
SparseMatrix<T> operator*(const T& scalar, const SparseMatrix<T>& matrix) {
    return matrix * scalar;
}

} // namespace linalg
} // namespace math
