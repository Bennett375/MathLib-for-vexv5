#pragma once

#include <random>

namespace math {
namespace random {

class RandomSource {
public:
    virtual ~RandomSource() {}

    // Uniform value in [0, 1).
    virtual double uniform01() = 0;

    // Sample from N(mean, standard_deviation^2).
    virtual double normal(
        double mean,
        double standard_deviation) = 0;
};

class Rng : public RandomSource {
public:
    Rng(int seed = 0);

    double uniform01() override;

    double normal(double mean, double standard_deviation) override;

private:
    std::mt19937 gen_;
};

} // namespace random
} // namespace math
