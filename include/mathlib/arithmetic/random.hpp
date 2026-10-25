#pragma once

#include <random>


namespace math {
namespace random {
class RandomSource {
private:
    std::mt19937 gen_;

    static std::mt19937 create_engine(int seed) {
        if (seed == 0) {
            std::random_device rd;
            return std::mt19937(rd());
        }
        return std::mt19937(seed);
    }
public:
    RandomSource(int seed = 0): gen_(create_engine(seed)) {}

    virtual ~RandomSource() {}

    // Uniform value in [0, 1).
    virtual double uniform01() = 0;

    // Sample from N(mean, standard_deviation^2).
    virtual double normal(
        double mean,
        double standard_deviation) = 0;
};  

class Rng : public RandomSource {
private:
    std::mt19937 gen_;

    static std::mt19937 create_engine(int seed) {
        if (seed == 0) {
            std::random_device rd;
            return std::mt19937(rd());
        }
        return std::mt19937(seed);
    }
public:    
    Rng(int seed = 0): gen_(create_engine(seed)) {}
    double uniform01() override;

    double normal(double mean, double standard_deviation) override;
};

}
}