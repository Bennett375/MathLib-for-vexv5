#include "random.hpp"


namespace math {
namespace random {

double Rng::uniform01() {
    std::uniform_real_distribution<double> dist(0, 1);
    return dist(gen_);
}

double Rng::normal(double mean, double standard_deviation) {
    std::normal_distribution<double> dist(mean, standard_deviation);
    return dist(gen_);
}

}
}