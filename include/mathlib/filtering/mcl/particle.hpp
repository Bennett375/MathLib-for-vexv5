#pragma once

#include "mathlib/linalg/vector.hpp"
#include "mathlib/abstract/object.hpp"

namespace math {
namespace filters {
namespace mcl {

struct Particle : MathObj<Particle> {
    linalg::Vector<double> state;
    double weight;

    Particle(std::size_t dim)
        : state(dim, 0), weight(0.0) {}

    Particle(const linalg::Vector<double>& x, double w)
        : state(x), weight(w) {}
};

}
}

template <>
struct conversion<filters::mcl::Particle, std::string> {
    static std::string convert(const filters::mcl::Particle& particle) {
        return "Particle: " +
            conversion<linalg::Vector<double>, std::string>::convert(particle.state) +
            " with weight: " + std::to_string(particle.weight);
    }
};
}