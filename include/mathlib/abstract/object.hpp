#pragma once

#include <string>
#include <type_traits>

namespace math {

template <typename Object, typename Target>
struct conversion; // Intentionally undefined.

template <typename Derived>
class MathObj {
public:
    template <typename T>
    T as() const {
        return conversion<Derived, T>::convert(
            static_cast<const Derived&>(*this)
        );
    }
};

}