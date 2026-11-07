#pragma once

#include "mathlib/abstract/object.hpp"
#include "mathlib/linalg/matrix_storage.hpp"
#include "mathlib/linalg/vector.hpp"

#include <cstddef>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <utility>

namespace math {
namespace linalg {

template <typename T>
class SparseVector : public MathObj<SparseVector<T>>,
                     public MatrixStorage<
                         T, detail::SparseVectorStorage<T>> {
public:
    typedef MatrixStorage<T, detail::SparseVectorStorage<T>> Base;
    typedef detail::SparseVectorStorage<T> Storage;

    SparseVector() : Base() {}

    explicit SparseVector(std::size_t size) : Base(size, 1) {}

    SparseVector(std::size_t size, const T& initial_value)
        : Base(size, 1) {
        if (initial_value != T()) {
            for (std::size_t i = 0; i < size; ++i) {
                this->set_storage_value(i, 0, initial_value);
            }
        }
    }

    SparseVector(std::initializer_list<T> values)
        : Base(values.size(), 1) {
        std::size_t index = 0;
        for (typename std::initializer_list<T>::const_iterator it =
                 values.begin();
             it != values.end(); ++it, ++index) {
            set(index, *it);
        }
    }

    explicit SparseVector(const Vector<T>& values)
        : Base(values.size(), 1) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            set(i, values(i));
        }
    }

    std::size_t size() const { return this->rows_; }
    bool empty() const { return this->rows_ == 0; }
    std::size_t non_zero_count() const { return this->data_.size(); }

    T operator()(std::size_t index) const {
        require_index(index);
        return this->storage_value(index, 0);
    }

    void set(std::size_t index, const T& value) {
        require_index(index);
        this->set_storage_value(index, 0, value);
    }

    void erase(std::size_t index) {
        require_index(index);
        this->data_.erase(index);
    }

    void clear() { this->data_.clear(); }

    void fill(const T& value) {
        this->fill_storage(value);
    }

    const Storage& data() const { return this->data_; }

    Vector<T> to_dense() const {
        Vector<T> result(size());
        for (typename Storage::const_iterator it = this->data_.begin();
             it != this->data_.end(); ++it) {
            result(it->first) = it->second;
        }
        return result;
    }

    double sum() const {
        return this->storage_sum();
    }

    SparseVector operator+(const SparseVector& other) const {
        require_same_size(other);
        SparseVector result(*this);
        for (typename Storage::const_iterator it = other.data_.begin();
             it != other.data_.end(); ++it) {
            result.set(it->first, result(it->first) + it->second);
        }
        return result;
    }

    SparseVector operator-(const SparseVector& other) const {
        require_same_size(other);
        SparseVector result(*this);
        for (typename Storage::const_iterator it = other.data_.begin();
             it != other.data_.end(); ++it) {
            result.set(it->first, result(it->first) - it->second);
        }
        return result;
    }

    SparseVector operator*(const T& scalar) const {
        SparseVector result(size());
        if (scalar == T()) {
            return result;
        }
        for (typename Storage::const_iterator it = this->data_.begin();
             it != this->data_.end(); ++it) {
            result.set(it->first, it->second * scalar);
        }
        return result;
    }

    SparseVector operator/(const T& scalar) const {
        if (scalar == T()) {
            throw std::invalid_argument(
                "Cannot divide a sparse vector by zero.");
        }
        SparseVector result(size());
        for (typename Storage::const_iterator it = this->data_.begin();
             it != this->data_.end(); ++it) {
            result.set(it->first, it->second / scalar);
        }
        return result;
    }

    SparseVector& operator+=(const SparseVector& other) {
        *this = *this + other;
        return *this;
    }

    SparseVector& operator-=(const SparseVector& other) {
        *this = *this - other;
        return *this;
    }

    SparseVector& operator*=(const T& scalar) {
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

    static SparseVector zeros(std::size_t size) {
        return SparseVector(size);
    }

private:
    using Base::rows_;
    using Base::data_;

    void require_index(std::size_t index) const {
        if (index >= this->rows_) {
            throw std::out_of_range("Sparse vector index out of range.");
        }
    }

    void require_same_size(const SparseVector& other) const {
        if (this->rows_ != other.rows_) {
            throw std::invalid_argument(
                "Sparse vector dimensions do not match.");
        }
    }
};

template <typename T>
double dot(const SparseVector<T>& left, const SparseVector<T>& right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("Sparse vector dimensions do not match.");
    }

    const typename SparseVector<T>::Storage* smaller = &left.data();
    const SparseVector<T>* other = &right;
    if (left.non_zero_count() > right.non_zero_count()) {
        smaller = &right.data();
        other = &left;
    }

    double result = 0.0;
    for (typename SparseVector<T>::Storage::const_iterator it =
             smaller->begin();
         it != smaller->end(); ++it) {
        result += static_cast<double>(it->second) *
                  static_cast<double>((*other)(it->first));
    }
    return result;
}

template <typename T>
SparseVector<T> operator*(const T& scalar, const SparseVector<T>& vector) {
    return vector * scalar;
}

} // namespace linalg
} // namespace math
