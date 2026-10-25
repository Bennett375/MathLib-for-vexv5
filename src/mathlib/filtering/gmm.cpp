
#include "gmm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace math {
namespace filters {

namespace {

typedef solver::SolveStatus Status;
typedef solver::SolveOptions<double> Options;

bool finite_matrix(const linalg::Matrix<double>& A) {
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            if (!std::isfinite(A(i, j))) {
                return false;
            }
        }
    }
    return true;
}

bool finite_vector(const linalg::Vector<double>& x) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x(i))) {
            return false;
        }
    }
    return true;
}

linalg::Matrix<double> outer_product(
    const linalg::Vector<double>& a,
    const linalg::Vector<double>& b) {

    linalg::Matrix<double> result(a.size(), b.size(), 0.0);

    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) {
            result(i, j) = a(i) * b(j);
        }
    }

    return result;
}

// Symmetrize a covariance and apply a minimum diagonal variance.
// This does not guarantee positive definiteness; Cholesky validation
// is still required.
Status regularize_covariance(
    linalg::Matrix<double>& covariance,
    double floor) {

    if (covariance.rows() != covariance.cols() ||
        !finite_matrix(covariance) ||
        !std::isfinite(floor) ||
        floor < 0.0) {
        return Status::NumericalFailure;
    }

    const std::size_t n = covariance.rows();

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double average =
                0.5 * covariance(i, j) +
                0.5 * covariance(j, i);

            covariance(i, j) = average;
            covariance(j, i) = average;
        }

        if (covariance(i, i) < floor) {
            covariance(i, i) = floor;
        }
    }

    return Status::Success;
}

Status validate_spd(
    const linalg::Matrix<double>& covariance) {

    solver::CholeskyResult<double> result =
        solver::cholesky_decompose(
            covariance, Options());

    return result.status;
}

// Solve A*x=b using the existing Cholesky solver.
// A must be symmetric positive definite.
Status solve_spd(
    const linalg::Matrix<double>& A,
    const linalg::Vector<double>& b,
    linalg::Vector<double>& x) {

    solver::SolveResult<double> result =
        solver::Solver<double>::solve(
            A,
            b,
            solver::SolveMethod::Cholesky,
            Options());

    if (!result.success()) {
        return result.status;
    }

    x = result.solution;

    if (!finite_vector(x)) {
        return Status::NumericalFailure;
    }

    return Status::Success;
}

// Compute K = P*H^T*S^-1 without explicitly inverting S.
// Each row of P*H^T is transposed and solved against S.
Status compute_kalman_gain(
    const linalg::Matrix<double>& P,
    const linalg::Matrix<double>& H,
    const linalg::Matrix<double>& S,
    linalg::Matrix<double>& K) {

    const linalg::Matrix<double> PHt = P * H.transpose();

    K = linalg::Matrix<double>(P.rows(), H.rows(), 0.0);

    for (std::size_t i = 0; i < PHt.rows(); ++i) {
        linalg::Vector<double> rhs(PHt.cols(), 0.0);

        for (std::size_t j = 0; j < PHt.cols(); ++j) {
            rhs(j) = PHt(i, j);
        }

        linalg::Vector<double> solution(PHt.cols(), 0.0);

        const Status status = solve_spd(S, rhs, solution);

        if (status != Status::Success) {
            return status;
        }

        for (std::size_t j = 0; j < solution.size(); ++j) {
            K(i, j) = solution(j);
        }
    }

    return finite_matrix(K)
        ? Status::Success
        : Status::NumericalFailure;
}

Status make_gaussian(
    const linalg::Vector<double>& mean,
    linalg::Matrix<double> covariance,
    double covariance_floor,
    linalg::Gaussian& output) {

    Status status =
        regularize_covariance(covariance, covariance_floor);

    if (status != Status::Success) {
        return status;
    }

    return linalg::Gaussian::create(mean, covariance, output);
}

// Return the Gaussian log likelihood for an innovation.
// A zero-mean Gaussian with covariance S evaluates
// the innovation's log likelihood.
Status innovation_log_likelihood(
    const linalg::Vector<double>& innovation,
    const linalg::Matrix<double>& S,
    double covariance_floor,
    double& log_likelihood) {

    linalg::Vector<double> zero_mean(innovation.size(), 0.0);
    linalg::Matrix<double> covariance = S;
    linalg::Gaussian likelihood;

    const Status status = make_gaussian(
        zero_mean, covariance, covariance_floor, likelihood);

    if (status != Status::Success) {
        return status;
    }

    const linalg::GaussianValueResult result =
        likelihood.log_pdf(innovation);

    if (!result.success()) {
        return result.status;
    }

    log_likelihood = result.value;
    return Status::Success;
}

double vector_squared_norm(const linalg::Vector<double>& x) {
    double result = 0.0;

    for (std::size_t i = 0; i < x.size(); ++i) {
        result += x(i) * x(i);
    }

    return result;
}

} // namespace


// -----------------------------------------------------------------------------
// Construction and initialization
// -----------------------------------------------------------------------------

GaussianMixtureFilter::GaussianMixtureFilter(
    const GMMOptions& options)
    : options_(options) {}

Status GaussianMixtureFilter::initialize(
    const std::vector<GaussianComponent>& components) {

    if (components.empty() ||
        options_.max_components == 0 ||
        !std::isfinite(options_.min_weight) ||
        options_.min_weight < 0.0 ||
        !std::isfinite(options_.merge_threshold) ||
        options_.merge_threshold < 0.0 ||
        !std::isfinite(options_.covariance_floor) ||
        options_.covariance_floor < 0.0) {
        return Status::InvalidDimensions;
    }

    const std::size_t dimension =
        components[0].distribution.dim();

    if (dimension == 0) {
        return Status::InvalidDimensions;
    }

    double weight_sum = 0.0;

    for (std::size_t i = 0; i < components.size(); ++i) {
        const GaussianComponent& component = components[i];

        if (!component.distribution.valid() ||
            component.distribution.dim() != dimension) {
            return Status::InvalidDimensions;
        }

        if (!std::isfinite(component.weight) ||
            component.weight < 0.0) {
            return Status::NumericalFailure;
        }

        weight_sum += component.weight;
    }

    if (!std::isfinite(weight_sum) || weight_sum <= 0.0) {
        return Status::NumericalFailure;
    }

    // Build the new mixture separately so a failed initialization
    // leaves the previous mixture intact.
    std::vector<GaussianComponent> candidate = components;

    for (std::size_t i = 0; i < candidate.size(); ++i) {
        candidate[i].weight /= weight_sum;
    }

    const std::vector<GaussianComponent> previous = components_;
    components_.swap(candidate);

    const Status status = prune();

    if (status != Status::Success) {
        components_ = previous;
        return status;
    }

    return Status::Success;
}


// -----------------------------------------------------------------------------
// Prediction
// x' = F*x
// P' = F*P*F^T + Q
// -----------------------------------------------------------------------------

Status GaussianMixtureFilter::predict(
    const linalg::Matrix<double>& F,
    const linalg::Matrix<double>& Q) {

    if (components_.empty()) {
        return Status::InvalidDimensions;
    }

    const std::size_t n =
        components_[0].distribution.dim();

    if (F.rows() != n || F.cols() != n ||
        Q.rows() != n || Q.cols() != n) {
        return Status::InvalidDimensions;
    }

    if (!finite_matrix(F) || !finite_matrix(Q)) {
        return Status::NonFiniteInput;
    }

    linalg::Matrix<double> process_noise = Q;

    Status status = regularize_covariance(
        process_noise, options_.covariance_floor);

    if (status != Status::Success) {
        return status;
    }

    status = validate_spd(process_noise);

    if (status != Status::Success) {
        return status;
    }

    const linalg::Matrix<double> Ft = F.transpose();

    std::vector<GaussianComponent> candidate;
    candidate.reserve(components_.size());

    for (std::size_t i = 0; i < components_.size(); ++i) {
        const GaussianComponent& current = components_[i];

        const linalg::Vector<double> predicted_mean =
            F * current.distribution.mean();

        const linalg::Matrix<double> FP =
            F * current.distribution.covariance();

        linalg::Matrix<double> predicted_covariance =
            (FP * Ft) + process_noise;

        linalg::Gaussian predicted;

        status = make_gaussian(
            predicted_mean,
            predicted_covariance,
            options_.covariance_floor,
            predicted);

        if (status != Status::Success) {
            return status;
        }

        GaussianComponent next;
        next.distribution = predicted;
        next.weight = current.weight;

        candidate.push_back(next);
    }

    components_.swap(candidate);
    return Status::Success;
}


// -----------------------------------------------------------------------------
// Measurement update
//
// Innovation: r = z - H*mu
// Innovation covariance: S = H*P*H^T + R
// Gain: K = P*H^T*S^-1
// Mean: mu' = mu + K*r
// Covariance uses Joseph form.
// -----------------------------------------------------------------------------

Status GaussianMixtureFilter::update(
    const linalg::Vector<double>& measurement,
    const linalg::Matrix<double>& H,
    const linalg::Matrix<double>& R) {

    if (components_.empty()) {
        return Status::InvalidDimensions;
    }

    const std::size_t n =
        components_[0].distribution.dim();
    const std::size_t m = measurement.size();

    if (m == 0 ||
        H.rows() != m ||
        H.cols() != n ||
        R.rows() != m ||
        R.cols() != m) {
        return Status::InvalidDimensions;
    }

    if (!finite_vector(measurement) ||
        !finite_matrix(H) ||
        !finite_matrix(R)) {
        return Status::NonFiniteInput;
    }

    linalg::Matrix<double> measurement_noise = R;

    Status status = regularize_covariance(
        measurement_noise, options_.covariance_floor);

    if (status != Status::Success) {
        return status;
    }

    status = validate_spd(measurement_noise);

    if (status != Status::Success) {
        return status;
    }

    const linalg::Matrix<double> Ht = H.transpose();
    const linalg::Matrix<double> I = linalg::Matrix<double>::identity(n);

    std::vector<GaussianComponent> candidate;
    candidate.reserve(components_.size());

    std::vector<double> log_weights;
    log_weights.reserve(components_.size());

    // Update every component into temporary storage. No changes are
    // committed unless all component updates succeed.
    for (std::size_t i = 0; i < components_.size(); ++i) {
        const GaussianComponent& current = components_[i];

        const linalg::Vector<double> predicted_measurement =
            H * current.distribution.mean();

        const linalg::Vector<double> innovation =
            measurement - predicted_measurement;

        const linalg::Matrix<double>& P =
            current.distribution.covariance();

        const linalg::Matrix<double> HP =
            H * P;

        linalg::Matrix<double> S =
            (HP * Ht) + measurement_noise;

        status = regularize_covariance(
            S, options_.covariance_floor);

        if (status != Status::Success) {
            return status;
        }

        status = validate_spd(S);

        if (status != Status::Success) {
            return status;
        }

        double log_likelihood = 0.0;

        status = innovation_log_likelihood(
            innovation,
            S,
            options_.covariance_floor,
            log_likelihood);

        if (status != Status::Success) {
            return status;
        }

        linalg::Matrix<double> K(n, m, 0.0);

        status = compute_kalman_gain(P, H, S, K);

        if (status != Status::Success) {
            return status;
        }

        const linalg::Vector<double> correction =
            K * innovation;

        const linalg::Vector<double> updated_mean =
            current.distribution.mean() + correction;

        // Joseph covariance update:
        // P' = (I-KH)P(I-KH)^T + KRK^T
        const linalg::Matrix<double> KH = K * H;
        const linalg::Matrix<double> A = I - KH;
        const linalg::Matrix<double> At = A.transpose();
        const linalg::Matrix<double> Kt = K.transpose();

        linalg::Matrix<double> updated_covariance =
            ((A * P) * At) + ((K * measurement_noise) * Kt);

        linalg::Gaussian updated;

        status = make_gaussian(
            updated_mean,
            updated_covariance,
            options_.covariance_floor,
            updated);

        if (status != Status::Success) {
            return status;
        }

        GaussianComponent next;
        next.distribution = updated;
        next.weight = current.weight;

        candidate.push_back(next);

        if (current.weight > 0.0) {
            log_weights.push_back(
                std::log(current.weight) + log_likelihood);
        } else {
            log_weights.push_back(
                -std::numeric_limits<double>::infinity());
        }
    }

    // Log-sum-exp normalization.
    double max_log_weight =
        -std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < log_weights.size(); ++i) {
        if (log_weights[i] > max_log_weight) {
            max_log_weight = log_weights[i];
        }
    }

    if (!std::isfinite(max_log_weight)) {
        return Status::NumericalFailure;
    }

    double total = 0.0;

    for (std::size_t i = 0; i < log_weights.size(); ++i) {
        candidate[i].weight =
            std::exp(log_weights[i] - max_log_weight);

        total += candidate[i].weight;
    }

    if (!std::isfinite(total) || total <= 0.0) {
        return Status::NumericalFailure;
    }

    for (std::size_t i = 0; i < candidate.size(); ++i) {
        candidate[i].weight /= total;
    }

    components_.swap(candidate);

    return prune();
}


// -----------------------------------------------------------------------------
// Weight normalization
// -----------------------------------------------------------------------------

void GaussianMixtureFilter::normalize_weights() {
    double total = 0.0;

    for (std::size_t i = 0; i < components_.size(); ++i) {
        total += components_[i].weight;
    }

    if (!std::isfinite(total) || total <= 0.0) {
        return;
    }

    for (std::size_t i = 0; i < components_.size(); ++i) {
        components_[i].weight /= total;
    }
}


// -----------------------------------------------------------------------------
// Merge similar components
// -----------------------------------------------------------------------------

Status GaussianMixtureFilter::merge_similar_components() {
    if (components_.size() < 2) {
        return Status::Success;
    }

    const double threshold = options_.merge_threshold;

    bool merged = true;

    while (merged) {
        merged = false;

        for (std::size_t i = 0;
             i < components_.size() && !merged;
             ++i) {

            for (std::size_t j = i + 1;
                 j < components_.size();
                 ++j) {

                const GaussianComponent& a = components_[i];
                const GaussianComponent& b = components_[j];

                if (a.weight <= 0.0 || b.weight <= 0.0) {
                    continue;
                }

                const linalg::GaussianValueResult d_ab =
                    a.distribution.mahalanobis_squared(
                        b.distribution.mean());

                const linalg::GaussianValueResult d_ba =
                    b.distribution.mahalanobis_squared(
                        a.distribution.mean());

                if (!d_ab.success()) {
                    return d_ab.status;
                }

                if (!d_ba.success()) {
                    return d_ba.status;
                }

                const double distance_squared =
                    0.5 * (d_ab.value + d_ba.value);

                if (!std::isfinite(distance_squared)) {
                    return Status::NumericalFailure;
                }

                if (distance_squared >= threshold) {
                    continue;
                }

                const double weight =
                    a.weight + b.weight;

                if (!std::isfinite(weight) || weight <= 0.0) {
                    return Status::NumericalFailure;
                }

                const double alpha = a.weight / weight;
                const double beta = b.weight / weight;

                const linalg::Vector<double>& mu_a =
                    a.distribution.mean();
                const linalg::Vector<double>& mu_b =
                    b.distribution.mean();

                const std::size_t n = mu_a.size();

                linalg::Vector<double> merged_mean(n, 0.0);

                for (std::size_t k = 0; k < n; ++k) {
                    merged_mean(k) =
                        alpha * mu_a(k) +
                        beta * mu_b(k);
                }

                const linalg::Vector<double> delta_a =
                    mu_a - merged_mean;

                const linalg::Vector<double> delta_b =
                    mu_b - merged_mean;

                linalg::Matrix<double> covariance_a =
                    a.distribution.covariance() +
                    outer_product(delta_a, delta_a);

                linalg::Matrix<double> covariance_b =
                    b.distribution.covariance() +
                    outer_product(delta_b, delta_b);

                linalg::Matrix<double> merged_covariance =
                    (covariance_a * alpha) +
                    (covariance_b * beta);

                linalg::Gaussian merged_distribution;

                const Status status = make_gaussian(
                    merged_mean,
                    merged_covariance,
                    options_.covariance_floor,
                    merged_distribution);

                if (status != Status::Success) {
                    return status;
                }

                GaussianComponent replacement;
                replacement.distribution = merged_distribution;
                replacement.weight = weight;

                components_[i] = replacement;

                components_.erase(
                    components_.begin() +
                    static_cast<std::ptrdiff_t>(j));

                normalize_weights();

                merged = true;
                break;
            }
        }
    }

    return Status::Success;
}


// -----------------------------------------------------------------------------
// Dynamic pruning
// -----------------------------------------------------------------------------

Status GaussianMixtureFilter::prune() {
    if (components_.empty()) {
        return Status::InvalidDimensions;
    }

    // Remove negligible weights.
    std::vector<GaussianComponent> retained;
    retained.reserve(components_.size());

    std::size_t best_index = 0;

    for (std::size_t i = 1; i < components_.size(); ++i) {
        if (components_[i].weight > components_[best_index].weight) {
            best_index = i;
        }
    }

    for (std::size_t i = 0; i < components_.size(); ++i) {
        if (components_[i].weight >= options_.min_weight) {
            retained.push_back(components_[i]);
        }
    }

    // Never discard every component solely due to the weight threshold.
    if (retained.empty()) {
        retained.push_back(components_[best_index]);
    }

    components_.swap(retained);
    normalize_weights();

    Status status = merge_similar_components();

    if (status != Status::Success) {
        return status;
    }

    // Enforce the maximum count by keeping the largest weights.
    if (components_.size() > options_.max_components) {
        std::sort(
            components_.begin(),
            components_.end(),
            [](const GaussianComponent& a,
               const GaussianComponent& b) {
                return a.weight > b.weight;
            });

        components_.resize(options_.max_components);
    }

    if (components_.empty()) {
        return Status::NumericalFailure;
    }

    normalize_weights();

    double sum = 0.0;

    for (std::size_t i = 0; i < components_.size(); ++i) {
        if (!std::isfinite(components_[i].weight) ||
            components_[i].weight < 0.0) {
            return Status::NumericalFailure;
        }

        sum += components_[i].weight;
    }

    if (!std::isfinite(sum) || sum <= 0.0) {
        return Status::NumericalFailure;
    }

    return Status::Success;
}


// -----------------------------------------------------------------------------
// Mixture estimate
//
// Mean = sum(w_i * mu_i)
//
// Covariance = sum(w_i * [P_i +
//              (mu_i - mean)(mu_i - mean)^T])
// -----------------------------------------------------------------------------

GMMEstimate GaussianMixtureFilter::estimate() const {
    if (components_.empty()) {
        GMMEstimate result(0);
        result.status = Status::InvalidDimensions;
        return result;
    }
    const std::size_t n =
        components_[0].distribution.dim();

    GMMEstimate result(n);

    double total_weight = 0.0;

    for (std::size_t i = 0; i < components_.size(); ++i) {
        total_weight += components_[i].weight;
    }

    if (!std::isfinite(total_weight) || total_weight <= 0.0) {
        result.status = Status::NumericalFailure;
        return result;
    }

    // Weighted mean.
    for (std::size_t i = 0; i < components_.size(); ++i) {
        const GaussianComponent& component = components_[i];
        const double weight = component.weight / total_weight;
        const linalg::Vector<double>& mu = component.distribution.mean();

        for (std::size_t j = 0; j < n; ++j) {
            result.mean(j) += weight * mu(j);
        }
    }

    // Total covariance, including between-component uncertainty.
    for (std::size_t i = 0; i < components_.size(); ++i) {
        const GaussianComponent& component = components_[i];
        const double weight = component.weight / total_weight;

        const linalg::Vector<double> delta =
            component.distribution.mean() - result.mean;

        const linalg::Matrix<double> contribution =
            component.distribution.covariance() +
            outer_product(delta, delta);

        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                result.covariance(row, col) +=
                    weight * contribution(row, col);
            }
        }
    }

    if (!finite_vector(result.mean) ||
        !finite_matrix(result.covariance)) {
        result.status = Status::NumericalFailure;
        return result;
    }

    result.status = Status::Success;
    return result;
}


// -----------------------------------------------------------------------------
// Accessors
// -----------------------------------------------------------------------------

std::size_t GaussianMixtureFilter::component_count() const {
    return components_.size();
}

const std::vector<GaussianComponent>&
GaussianMixtureFilter::components() const {
    return components_;
}

const GaussianComponent&
GaussianMixtureFilter::most_likely_component() const {
    // A static invalid component makes the empty case safe to reference.
    static const GaussianComponent empty_component;

    if (components_.empty()) {
        return empty_component;
    }

    std::size_t best = 0;

    for (std::size_t i = 1; i < components_.size(); ++i) {
        if (components_[i].weight > components_[best].weight) {
            best = i;
        }
    }

    return components_[best];
}

} // namespace filters
} // namespace math
