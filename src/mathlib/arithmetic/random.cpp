#include "mathlib/arithmetic/random.hpp"

namespace math {
namespace random {
namespace {

std::mt19937 create_engine(int seed) {
    if (seed == 0) {
        std::random_device rd;
        return std::mt19937(rd());
    }

    return std::mt19937(seed);
}

} // namespace

Rng::Rng(int seed)
    : gen_(create_engine(seed)) {}

double Rng::uniform01() {
    std::uniform_real_distribution<double> dist(0, 1);
    return dist(gen_);
}

double Rng::normal(double mean, double standard_deviation) {
    std::normal_distribution<double> dist(mean, standard_deviation);
    return dist(gen_);
}

} // namespace random
} // namespace math
