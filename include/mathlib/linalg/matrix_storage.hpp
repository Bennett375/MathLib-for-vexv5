#pragma once

#include <cstddef>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace math {
namespace linalg {
namespace detail {

template <typename T>
class DenseMatrixStorage : public std::vector<T> {
public:
    DenseMatrixStorage(
        std::size_t rows,
        std::size_t cols,
        const T& initial_value)
        : std::vector<T>(rows * cols, initial_value) {}

    T value(std::size_t row, std::size_t col, std::size_t cols) const {
        return (*this)[row * cols + col];
    }

    T& reference(std::size_t row, std::size_t col, std::size_t cols) {
        return (*this)[row * cols + col];
    }

    void set(
        std::size_t row,
        std::size_t col,
        std::size_t cols,
        const T& value) {
        (*this)[row * cols + col] = value;
    }

    void fill(std::size_t, std::size_t, const T& value) {
        std::fill(this->begin(), this->end(), value);
    }

    double sum() const {
        double result = 0.0;
        for (typename std::vector<T>::const_iterator it = this->begin();
             it != this->end(); ++it) {
            result += static_cast<double>(*it);
        }
        return result;
    }
};

template <typename T>
class SparseMatrixStorage : public std::map<
    std::pair<std::size_t, std::size_t>, T> {
public:
    typedef std::map<std::pair<std::size_t, std::size_t>, T> Map;
    typedef typename Map::key_type Key;

    SparseMatrixStorage() : Map() {}

    SparseMatrixStorage(
        std::size_t,
        std::size_t,
        const T&) : Map() {}

    T value(std::size_t row, std::size_t col, std::size_t) const {
        typename Map::const_iterator it = this->find(Key(row, col));
        return it == this->end() ? T() : it->second;
    }

    void set(
        std::size_t row,
        std::size_t col,
        std::size_t,
        const T& value) {
        const Key key(row, col);
        if (value == T()) {
            this->erase(key);
        } else {
            (*this)[key] = value;
        }
    }

    void fill(std::size_t rows, std::size_t cols, const T& value) {
        this->clear();
        if (value == T()) {
            return;
        }
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                (*this)[Key(row, col)] = value;
            }
        }
    }

    double sum() const {
        double result = 0.0;
        for (typename Map::const_iterator it = this->begin();
             it != this->end(); ++it) {
            result += static_cast<double>(it->second);
        }
        return result;
    }
};

template <typename T>
class SparseVectorStorage : public std::map<std::size_t, T> {
public:
    typedef std::map<std::size_t, T> Map;

    SparseVectorStorage() : Map() {}

    SparseVectorStorage(
        std::size_t,
        std::size_t,
        const T&) : Map() {}

    T value(std::size_t row, std::size_t col, std::size_t) const {
        const std::size_t index = row + col;
        typename Map::const_iterator it = this->find(index);
        return it == this->end() ? T() : it->second;
    }

    void set(
        std::size_t row,
        std::size_t col,
        std::size_t,
        const T& value) {
        const std::size_t index = row + col;
        if (value == T()) {
            this->erase(index);
        } else {
            (*this)[index] = value;
        }
    }

    void fill(std::size_t rows, std::size_t cols, const T& value) {
        this->clear();
        if (value == T()) {
            return;
        }
        for (std::size_t i = 0; i < rows * cols; ++i) {
            (*this)[i] = value;
        }
    }

    double sum() const {
        double result = 0.0;
        for (typename Map::const_iterator it = this->begin();
             it != this->end(); ++it) {
            result += static_cast<double>(it->second);
        }
        return result;
    }
};

} // namespace detail

template <typename T, typename Storage>
class MatrixStorage {
protected:
    MatrixStorage()
        : rows_(0), cols_(0), data_(0, 0, T()) {}

    MatrixStorage(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows, cols, T()) {}

    MatrixStorage(
        std::size_t rows,
        std::size_t cols,
        const T& initial_value)
        : rows_(rows), cols_(cols), data_(rows, cols, initial_value) {}

    T storage_value(std::size_t row, std::size_t col) const {
        validate_storage_index(row, col);
        return data_.value(row, col, cols_);
    }

    T& storage_reference(std::size_t row, std::size_t col) {
        validate_storage_index(row, col);
        return data_.reference(row, col, cols_);
    }

    void set_storage_value(
        std::size_t row,
        std::size_t col,
        const T& value) {
        validate_storage_index(row, col);
        data_.set(row, col, cols_, value);
    }

    void fill_storage(const T& value) {
        data_.fill(rows_, cols_, value);
    }

    double storage_sum() const {
        return data_.sum();
    }

    void validate_storage_index(std::size_t row, std::size_t col) const {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Matrix index out of range.");
        }
    }

    std::size_t rows_;
    std::size_t cols_;
    Storage data_;
};

} // namespace linalg
} // namespace math
