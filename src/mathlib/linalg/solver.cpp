#include "mathlib/linalg/solve_models.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace math {
namespace solver {

namespace detail {

// -----------------------------------------------------------------------------
// Internal numerical helpers
// Use double precision for intermediate calculations.
// -----------------------------------------------------------------------------

template <typename T>
double value_as_double(const T& value) {
    return static_cast<double>(value);
}

template <typename T>
linalg::Matrix<double> to_double_matrix(
    const linalg::Matrix<T>& A) {

    linalg::Matrix<double> result(A.rows(), A.cols());

    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            result(i, j) = value_as_double(A(i, j));
        }
    }

    return result;
}

template <typename T>
linalg::Vector<double> to_double_vector(
    const linalg::Vector<T>& b) {

    linalg::Vector<double> result(b.size());

    for (std::size_t i = 0; i < b.size(); ++i) {
        result(i) = value_as_double(b(i));
    }

    return result;
}

template <typename T>
linalg::Vector<T> to_scalar_vector(
    const linalg::Vector<double>& b) {

    linalg::Vector<T> result(b.size());

    for (std::size_t i = 0; i < b.size(); ++i) {
        result(i) = static_cast<T>(b(i));
    }

    return result;
}

template <typename T>
bool matrix_is_finite(const linalg::Matrix<T>& A) {
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            if (!std::isfinite(value_as_double(A(i, j)))) {
                return false;
            }
        }
    }

    return true;
}

template <typename T>
bool vector_is_finite(const linalg::Vector<T>& b) {
    for (std::size_t i = 0; i < b.size(); ++i) {
        if (!std::isfinite(value_as_double(b(i)))) {
            return false;
        }
    }

    return true;
}

inline double vector_norm(
    const linalg::Vector<double>& v) {

    double sum = 0.0;

    for (std::size_t i = 0; i < v.size(); ++i) {
        sum += v(i) * v(i);
    }

    return std::sqrt(sum);
}

inline double dot_double(
    const linalg::Vector<double>& a,
    const linalg::Vector<double>& b) {

    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector size mismatch.");
    }

    double sum = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a(i) * b(i);
    }

    return sum;
}

inline linalg::Vector<double> matrix_vector_product(
    const linalg::Matrix<double>& A,
    const linalg::Vector<double>& x) {

    if (A.cols() != x.size()) {
        throw std::invalid_argument(
            "Matrix and vector dimensions do not match.");
    }

    linalg::Vector<double> result(A.rows(), 0.0);

    for (std::size_t i = 0; i < A.rows(); ++i) {
        double sum = 0.0;

        for (std::size_t j = 0; j < A.cols(); ++j) {
            sum += A(i, j) * x(j);
        }

        result(i) = sum;
    }

    return result;
}

template <typename T>
SolveResult<T> make_failure(SolveStatus status) {
    SolveResult<T> result;
    result.status = status;
    return result;
}

} // namespace detail

// -----------------------------------------------------------------------------
// LU decomposition
//
// PA = LU
// permutation[i] identifies the original row occupying row i of PA.
// -----------------------------------------------------------------------------

template <typename T>
LUResult<T> lu_decompose(
    const linalg::Matrix<T>& input,
    const SolveOptions<T>& options) {

    LUResult<T> result;

    if (input.rows() != input.cols()) {
        result.status = SolveStatus::InvalidDimensions;
        return result;
    }

    if (options.check_finite &&
        !detail::matrix_is_finite(input)) {
        result.status = SolveStatus::NonFiniteInput;
        return result;
    }

    const std::size_t n = input.rows();
    linalg::Matrix<double> A = detail::to_double_matrix(input);

    linalg::Matrix<double> L(n, n, 0.0);
    linalg::Matrix<double> U = A;

    std::vector<std::size_t> permutation(n);

    for (std::size_t i = 0; i < n; ++i) {
        permutation[i] = i;
        L(i, i) = 1.0;
    }

    int sign = 1;
    double scale = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            scale = std::max(scale, std::fabs(A(i, j)));
        }
    }

    const double threshold =
        std::fabs(static_cast<double>(options.pivot_tolerance)) *
        std::max(1.0, scale);

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot_row = k;
        double pivot_abs = std::fabs(U(k, k));

        for (std::size_t i = k + 1; i < n; ++i) {
            const double candidate = std::fabs(U(i, k));

            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }

        if (pivot_abs <= threshold) {
            result.status = SolveStatus::SingularMatrix;
            return result;
        }

        if (pivot_row != k) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(U(k, j), U(pivot_row, j));
            }

            // Previously computed L columns must follow the row swap.
            for (std::size_t j = 0; j < k; ++j) {
                std::swap(L(k, j), L(pivot_row, j));
            }

            std::swap(permutation[k], permutation[pivot_row]);
            sign = -sign;
        }

        for (std::size_t i = k + 1; i < n; ++i) {
            const double factor = U(i, k) / U(k, k);
            L(i, k) = factor;
            U(i, k) = 0.0;

            for (std::size_t j = k + 1; j < n; ++j) {
                U(i, j) -= factor * U(k, j);
            }
        }
    }

    result.L = linalg::Matrix<T>(n, n);
    result.U = linalg::Matrix<T>(n, n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result.L(i, j) = static_cast<T>(L(i, j));
            result.U(i, j) = static_cast<T>(U(i, j));
        }
    }

    result.permutation = permutation;
    result.permutation_sign = sign;
    result.status = SolveStatus::Success;

    return result;
}

// -----------------------------------------------------------------------------
// Solve using LU factors
// -----------------------------------------------------------------------------

template <typename T>
SolveResult<T> solve_lu(
    const LUResult<T>& decomposition,
    const linalg::Vector<T>& b,
    const SolveOptions<T>& options) {

    if (!decomposition.success()) {
        return detail::make_failure<T>(decomposition.status);
    }

    const std::size_t n = decomposition.L.rows();

    if (decomposition.L.cols() != n ||
        decomposition.U.rows() != n ||
        decomposition.U.cols() != n ||
        b.size() != n ||
        decomposition.permutation.size() != n) {
        return detail::make_failure<T>(
            SolveStatus::InvalidDimensions);
    }

    if (options.check_finite &&
        (!detail::matrix_is_finite(decomposition.L) ||
         !detail::matrix_is_finite(decomposition.U) ||
         !detail::vector_is_finite(b))) {
        return detail::make_failure<T>(
            SolveStatus::NonFiniteInput);
    }

    linalg::Vector<double> y(n, 0.0);
    linalg::Vector<double> x(n, 0.0);

    // Forward substitution: Ly = Pb.
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t source_row = decomposition.permutation[i];

        if (source_row >= n) {
            return detail::make_failure<T>(
                SolveStatus::NumericalFailure);
        }

        double sum = static_cast<double>(b(source_row));

        for (std::size_t j = 0; j < i; ++j) {
            sum -= static_cast<double>(decomposition.L(i, j)) *
                   y(j);
        }

        const double diagonal =
            static_cast<double>(decomposition.L(i, i));

        if (std::fabs(diagonal) <=
            std::fabs(static_cast<double>(options.pivot_tolerance))) {
            return detail::make_failure<T>(
                SolveStatus::SingularMatrix);
        }

        y(i) = sum / diagonal;
    }

    // Back substitution: Ux = y.
    for (std::size_t reverse = 0; reverse < n; ++reverse) {
        const std::size_t i = n - 1 - reverse;
        double sum = y(i);

        for (std::size_t j = i + 1; j < n; ++j) {
            sum -= static_cast<double>(decomposition.U(i, j)) *
                   x(j);
        }

        const double diagonal =
            static_cast<double>(decomposition.U(i, i));

        if (std::fabs(diagonal) <=
            std::fabs(static_cast<double>(options.pivot_tolerance))) {
            return detail::make_failure<T>(
                SolveStatus::SingularMatrix);
        }

        x(i) = sum / diagonal;
    }

    SolveResult<T> result;
    result.solution = detail::to_scalar_vector<T>(x);
    result.status = SolveStatus::Success;
    result.iterations = 1;
    return result;
}

// -----------------------------------------------------------------------------
// Cholesky decomposition
//
// A = LL^T. Requires a symmetric positive-definite matrix.
// -----------------------------------------------------------------------------

template <typename T>
CholeskyResult<T> cholesky_decompose(
    const linalg::Matrix<T>& input,
    const SolveOptions<T>& options) {

    CholeskyResult<T> result;

    if (input.rows() != input.cols()) {
        result.status = SolveStatus::InvalidDimensions;
        return result;
    }

    if (options.check_finite &&
        !detail::matrix_is_finite(input)) {
        result.status = SolveStatus::NonFiniteInput;
        return result;
    }

    const std::size_t n = input.rows();
    linalg::Matrix<double> A = detail::to_double_matrix(input);
    linalg::Matrix<double> L(n, n, 0.0);

    double scale = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            scale = std::max(scale, std::fabs(A(i, j)));
        }
    }

    const double symmetry_tolerance =
        std::max(
            std::fabs(static_cast<double>(options.tolerance)),
            32.0 * std::numeric_limits<double>::epsilon()) *
        std::max(1.0, scale);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (std::fabs(A(i, j) - A(j, i)) >
                symmetry_tolerance) {
                result.status = SolveStatus::NotPositiveDefinite;
                return result;
            }
        }
    }

    const double positive_threshold =
        std::fabs(static_cast<double>(options.pivot_tolerance)) *
        std::max(1.0, scale);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = A(i, j);

            for (std::size_t k = 0; k < j; ++k) {
                sum -= L(i, k) * L(j, k);
            }

            if (i == j) {
                if (sum <= positive_threshold) {
                    result.status = SolveStatus::NotPositiveDefinite;
                    return result;
                }

                L(i, j) = std::sqrt(sum);
            } else {
                if (L(j, j) == 0.0) {
                    result.status = SolveStatus::NotPositiveDefinite;
                    return result;
                }

                L(i, j) = sum / L(j, j);
            }
        }
    }

    result.L = linalg::Matrix<T>(n, n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result.L(i, j) = static_cast<T>(L(i, j));
        }
    }

    result.status = SolveStatus::Success;
    return result;
}

// -----------------------------------------------------------------------------
// Solve using Cholesky factors
// -----------------------------------------------------------------------------

template <typename T>
SolveResult<T> solve_cholesky(
    const CholeskyResult<T>& decomposition,
    const linalg::Vector<T>& b,
    const SolveOptions<T>& options) {

    if (!decomposition.success()) {
        return detail::make_failure<T>(decomposition.status);
    }

    const std::size_t n = decomposition.L.rows();

    if (decomposition.L.cols() != n || b.size() != n) {
        return detail::make_failure<T>(
            SolveStatus::InvalidDimensions);
    }

    if (options.check_finite &&
        (!detail::matrix_is_finite(decomposition.L) ||
         !detail::vector_is_finite(b))) {
        return detail::make_failure<T>(
            SolveStatus::NonFiniteInput);
    }

    linalg::Vector<double> y(n, 0.0);
    linalg::Vector<double> x(n, 0.0);

    // Forward substitution: Ly = b.
    for (std::size_t i = 0; i < n; ++i) {
        double sum = static_cast<double>(b(i));

        for (std::size_t j = 0; j < i; ++j) {
            sum -= static_cast<double>(decomposition.L(i, j)) *
                   y(j);
        }

        const double diagonal =
            static_cast<double>(decomposition.L(i, i));

        if (diagonal == 0.0) {
            return detail::make_failure<T>(
                SolveStatus::NumericalFailure);
        }

        y(i) = sum / diagonal;
    }

    // Back substitution: L^T x = y.
    for (std::size_t reverse = 0; reverse < n; ++reverse) {
        const std::size_t i = n - 1 - reverse;
        double sum = y(i);

        for (std::size_t j = i + 1; j < n; ++j) {
            sum -= static_cast<double>(decomposition.L(j, i)) *
                   x(j);
        }

        const double diagonal =
            static_cast<double>(decomposition.L(i, i));

        if (diagonal == 0.0) {
            return detail::make_failure<T>(
                SolveStatus::NumericalFailure);
        }

        x(i) = sum / diagonal;
    }

    SolveResult<T> result;
    result.solution = detail::to_scalar_vector<T>(x);
    result.status = SolveStatus::Success;
    result.iterations = 1;
    return result;
}

// -----------------------------------------------------------------------------
// QR decomposition using modified Gram-Schmidt.
//
// Q is m x n and R is n x n. This implementation requires m >= n.
// -----------------------------------------------------------------------------

template <typename T>
QRResult<T> qr_decompose(
    const linalg::Matrix<T>& input,
    const SolveOptions<T>& options) {

    QRResult<T> result;

    const std::size_t m = input.rows();
    const std::size_t n = input.cols();

    if (m < n) {
        result.status = SolveStatus::InvalidDimensions;
        return result;
    }

    if (options.check_finite &&
        !detail::matrix_is_finite(input)) {
        result.status = SolveStatus::NonFiniteInput;
        return result;
    }

    linalg::Matrix<double> A = detail::to_double_matrix(input);
    linalg::Matrix<double> Q(m, n, 0.0);
    linalg::Matrix<double> R(n, n, 0.0);

    double scale = 0.0;

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            scale = std::max(scale, std::fabs(A(i, j)));
        }
    }

    const double threshold =
        std::fabs(static_cast<double>(options.pivot_tolerance)) *
        std::max(1.0, scale);

    std::size_t rank = 0;

    for (std::size_t j = 0; j < n; ++j) {
        linalg::Vector<double> v(m);

        for (std::size_t i = 0; i < m; ++i) {
            v(i) = A(i, j);
        }

        // Orthogonalize against previously computed Q columns.
        for (std::size_t k = 0; k < j; ++k) {
            double projection = 0.0;

            for (std::size_t i = 0; i < m; ++i) {
                projection += Q(i, k) * v(i);
            }

            R(k, j) += projection;

            for (std::size_t i = 0; i < m; ++i) {
                v(i) -= projection * Q(i, k);
            }
        }

        // A second pass improves orthogonality.
        for (std::size_t k = 0; k < j; ++k) {
            double correction = 0.0;

            for (std::size_t i = 0; i < m; ++i) {
                correction += Q(i, k) * v(i);
            }

            R(k, j) += correction;

            for (std::size_t i = 0; i < m; ++i) {
                v(i) -= correction * Q(i, k);
            }
        }

        const double magnitude = detail::vector_norm(v);
        R(j, j) = magnitude;

        if (magnitude <= threshold) {
            // Rank deficient: Q's j-th column remains zero.
            continue;
        }

        ++rank;

        for (std::size_t i = 0; i < m; ++i) {
            Q(i, j) = v(i) / magnitude;
        }
    }

    result.Q = linalg::Matrix<T>(m, n);
    result.R = linalg::Matrix<T>(n, n);

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result.Q(i, j) = static_cast<T>(Q(i, j));
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result.R(i, j) = static_cast<T>(R(i, j));
        }
    }

    result.rank = rank;
    result.status = SolveStatus::Success;
    return result;
}

// -----------------------------------------------------------------------------
// Least-squares solve using QR
// Minimizes ||Ax - b||_2 when A has full column rank.
// -----------------------------------------------------------------------------

template <typename T>
SolveResult<T> solve_qr(
    const QRResult<T>& decomposition,
    const linalg::Vector<T>& b,
    const SolveOptions<T>& options) {

    if (!decomposition.success()) {
        return detail::make_failure<T>(decomposition.status);
    }

    const std::size_t m = decomposition.Q.rows();
    const std::size_t n = decomposition.Q.cols();

    if (b.size() != m ||
        decomposition.R.rows() != n ||
        decomposition.R.cols() != n) {
        return detail::make_failure<T>(
            SolveStatus::InvalidDimensions);
    }

    if (decomposition.rank < n) {
        return detail::make_failure<T>(
            SolveStatus::SingularMatrix);
    }

    if (options.check_finite &&
        (!detail::matrix_is_finite(decomposition.Q) ||
         !detail::matrix_is_finite(decomposition.R) ||
         !detail::vector_is_finite(b))) {
        return detail::make_failure<T>(
            SolveStatus::NonFiniteInput);
    }

    linalg::Vector<double> y(n, 0.0);
    linalg::Vector<double> x(n, 0.0);

    // y = Q^T b.
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            y(j) +=
                static_cast<double>(decomposition.Q(i, j)) *
                static_cast<double>(b(i));
        }
    }

    // Back substitution: Rx = y.
    for (std::size_t reverse = 0; reverse < n; ++reverse) {
        const std::size_t i = n - 1 - reverse;
        double sum = y(i);

        for (std::size_t j = i + 1; j < n; ++j) {
            sum -= static_cast<double>(decomposition.R(i, j)) *
                   x(j);
        }

        const double diagonal =
            static_cast<double>(decomposition.R(i, i));

        if (std::fabs(diagonal) <=
            std::fabs(static_cast<double>(options.pivot_tolerance))) {
            return detail::make_failure<T>(
                SolveStatus::SingularMatrix);
        }

        x(i) = sum / diagonal;
    }

    SolveResult<T> result;
    result.solution = detail::to_scalar_vector<T>(x);
    result.status = SolveStatus::Success;
    result.iterations = 1;
    return result;
}

// -----------------------------------------------------------------------------
// High-level vector solve
// -----------------------------------------------------------------------------

template <typename T>
SolveResult<T> Solver<T>::solve(
    const linalg::Matrix<T>& A,
    const linalg::Vector<T>& b,
    SolveMethod method,
    const SolveOptions<T>& options) {

    if (A.rows() != b.size()) {
        return detail::make_failure<T>(
            SolveStatus::InvalidDimensions);
    }

    if (options.check_finite &&
        (!detail::matrix_is_finite(A) ||
         !detail::vector_is_finite(b))) {
        return detail::make_failure<T>(
            SolveStatus::NonFiniteInput);
    }

    if (method == SolveMethod::Automatic) {
        method = SolveMethod::LU;
    }

    if (method == SolveMethod::LU ||
        method == SolveMethod::GaussianElimination) {

        if (A.rows() != A.cols()) {
            return detail::make_failure<T>(
                SolveStatus::InvalidDimensions);
        }

        const LUResult<T> factors = lu_decompose(A, options);
        SolveResult<T> result = solve_lu(factors, b, options);

        if (result.success()) {
            result.residual_norm =
                residual_norm(A, result.solution, b);
        }

        return result;
    }

    if (method == SolveMethod::Cholesky) {
        if (A.rows() != A.cols()) {
            return detail::make_failure<T>(
                SolveStatus::InvalidDimensions);
        }

        const CholeskyResult<T> factors =
            cholesky_decompose(A, options);

        SolveResult<T> result =
            solve_cholesky(factors, b, options);

        if (result.success()) {
            result.residual_norm =
                residual_norm(A, result.solution, b);
        }

        return result;
    }

    if (method == SolveMethod::QR) {
        const QRResult<T> factors = qr_decompose(A, options);
        SolveResult<T> result = solve_qr(factors, b, options);

        if (result.success()) {
            result.residual_norm =
                residual_norm(A, result.solution, b);
        }

        return result;
    }

    if (method == SolveMethod::ConjugateGradient) {
        if (A.rows() != A.cols()) {
            return detail::make_failure<T>(
                SolveStatus::InvalidDimensions);
        }

        const std::size_t n = A.rows();
        const linalg::Matrix<double> Ad =
            detail::to_double_matrix(A);
        const linalg::Vector<double> bd =
            detail::to_double_vector(b);

        linalg::Vector<double> x(n);
        linalg::Vector<double> r = bd;
        linalg::Vector<double> p = r;

        double rr = detail::dot_double(r, r);
        const double initial_norm = std::sqrt(rr);
        const double tolerance =
            std::fabs(static_cast<double>(options.tolerance));

        SolveResult<T> result;

        if (initial_norm <= tolerance) {
            result.solution = detail::to_scalar_vector<T>(x);
            result.status = SolveStatus::Success;
            result.residual_norm = initial_norm;
            result.iterations = 0;
            return result;
        }

        for (std::size_t iteration = 0;
             iteration < options.max_iterations;
             ++iteration) {

            const linalg::Vector<double> Ap =
                detail::matrix_vector_product(Ad, p);

            const double denominator =
                detail::dot_double(p, Ap);

            if (denominator <= 0.0 ||
                !std::isfinite(denominator)) {
                return detail::make_failure<T>(
                    SolveStatus::NumericalFailure);
            }

            const double alpha = rr / denominator;

            for (std::size_t i = 0; i < n; ++i) {
                x(i) += alpha * p(i);
                r(i) -= alpha * Ap(i);
            }

            const double rr_new = detail::dot_double(r, r);
            const double residual = std::sqrt(rr_new);

            if (residual <= tolerance) {
                result.solution = detail::to_scalar_vector<T>(x);
                result.status = SolveStatus::Success;
                result.residual_norm = residual;
                result.iterations = iteration + 1;
                return result;
            }

            const double beta = rr_new / rr;

            for (std::size_t i = 0; i < n; ++i) {
                p(i) = r(i) + beta * p(i);
            }

            rr = rr_new;
        }

        result.solution = detail::to_scalar_vector<T>(x);
        result.status = SolveStatus::NoConvergence;
        result.residual_norm = std::sqrt(rr);
        result.iterations = options.max_iterations;
        return result;
    }

    return detail::make_failure<T>(SolveStatus::UnsupportedMethod);
}

// -----------------------------------------------------------------------------
// Solve multiple right-hand sides: AX = B
// Throws if the selected solver cannot solve the system.
// -----------------------------------------------------------------------------

template <typename T>
linalg::Matrix<T> Solver<T>::solve(
    const linalg::Matrix<T>& A,
    const linalg::Matrix<T>& B,
    SolveMethod method,
    const SolveOptions<T>& options) {

    if (A.rows() != B.rows()) {
        throw std::invalid_argument(
            "A and B must have the same number of rows.");
    }

    linalg::Matrix<T> X(A.cols(), B.cols());

    for (std::size_t col = 0; col < B.cols(); ++col) {
        linalg::Vector<T> rhs(B.rows());

        for (std::size_t row = 0; row < B.rows(); ++row) {
            rhs(row) = B(row, col);
        }

        const SolveResult<T> result =
            Solver<T>::solve(A, rhs, method, options);

        if (!result.success()) {
            throw std::runtime_error(
                "Matrix solve failed; inspect the vector solve result "
                "for the numerical status.");
        }

        for (std::size_t row = 0; row < X.rows(); ++row) {
            X(row, col) = result.solution(row);
        }
    }

    return X;
}

// -----------------------------------------------------------------------------
// Determinant using LU
// -----------------------------------------------------------------------------

template <typename T>
T determinant(
    const linalg::Matrix<T>& A,
    const SolveOptions<T>& options) {

    if (A.rows() != A.cols()) {
        throw std::invalid_argument(
            "Determinant requires a square matrix.");
    }

    const LUResult<T> factors = lu_decompose(A, options);

    if (!factors.success()) {
        return T();
    }

    double det = static_cast<double>(factors.permutation_sign);

    for (std::size_t i = 0; i < A.rows(); ++i) {
        det *= static_cast<double>(factors.U(i, i));
    }

    return static_cast<T>(det);
}

// -----------------------------------------------------------------------------
// Numerical rank using QR
// -----------------------------------------------------------------------------

template <typename T>
std::size_t numerical_rank(
    const linalg::Matrix<T>& A,
    const SolveOptions<T>& options) {

    const QRResult<T> factors = qr_decompose(A, options);

    if (!factors.success()) {
        return 0;
    }

    return factors.rank;
}

// -----------------------------------------------------------------------------
// Residual norm: ||Ax - b||_2
// -----------------------------------------------------------------------------

template <typename T>
double residual_norm(
    const linalg::Matrix<T>& A,
    const linalg::Vector<T>& x,
    const linalg::Vector<T>& b) {

    if (A.cols() != x.size() || A.rows() != b.size()) {
        throw std::invalid_argument(
            "Dimensions are incompatible for residual calculation.");
    }

    double sum = 0.0;

    for (std::size_t i = 0; i < A.rows(); ++i) {
        double predicted = 0.0;

        for (std::size_t j = 0; j < A.cols(); ++j) {
            predicted +=
                static_cast<double>(A(i, j)) *
                static_cast<double>(x(j));
        }

        const double error =
            predicted - static_cast<double>(b(i));

        sum += error * error;
    }

    return std::sqrt(sum);
}

// -----------------------------------------------------------------------------
// Explicit instantiations
// -----------------------------------------------------------------------------

// Explicit instantiations: int
template LUResult<int> lu_decompose<int>(
    const linalg::Matrix<int>&, const SolveOptions<int>&);
template CholeskyResult<int> cholesky_decompose<int>(
    const linalg::Matrix<int>&, const SolveOptions<int>&);
template QRResult<int> qr_decompose<int>(
    const linalg::Matrix<int>&, const SolveOptions<int>&);

template SolveResult<int> solve_lu<int>(
    const LUResult<int>&, const linalg::Vector<int>&,
    const SolveOptions<int>&);
template SolveResult<int> solve_cholesky<int>(
    const CholeskyResult<int>&, const linalg::Vector<int>&,
    const SolveOptions<int>&);
template SolveResult<int> solve_qr<int>(
    const QRResult<int>&, const linalg::Vector<int>&,
    const SolveOptions<int>&);

template class Solver<int>;

template int determinant<int>(
    const linalg::Matrix<int>&, const SolveOptions<int>&);
template std::size_t numerical_rank<int>(
    const linalg::Matrix<int>&, const SolveOptions<int>&);
template double residual_norm<int>(
    const linalg::Matrix<int>&,
    const linalg::Vector<int>&,
    const linalg::Vector<int>&);

// Explicit instantiations: float
template LUResult<float> lu_decompose<float>(
    const linalg::Matrix<float>&, const SolveOptions<float>&);
template CholeskyResult<float> cholesky_decompose<float>(
    const linalg::Matrix<float>&, const SolveOptions<float>&);
template QRResult<float> qr_decompose<float>(
    const linalg::Matrix<float>&, const SolveOptions<float>&);

template SolveResult<float> solve_lu<float>(
    const LUResult<float>&, const linalg::Vector<float>&,
    const SolveOptions<float>&);
template SolveResult<float> solve_cholesky<float>(
    const CholeskyResult<float>&, const linalg::Vector<float>&,
    const SolveOptions<float>&);
template SolveResult<float> solve_qr<float>(
    const QRResult<float>&, const linalg::Vector<float>&,
    const SolveOptions<float>&);

template class Solver<float>;

template float determinant<float>(
    const linalg::Matrix<float>&, const SolveOptions<float>&);
template std::size_t numerical_rank<float>(
    const linalg::Matrix<float>&, const SolveOptions<float>&);
template double residual_norm<float>(
    const linalg::Matrix<float>&,
    const linalg::Vector<float>&,
    const linalg::Vector<float>&);

// Explicit instantiations: double
template LUResult<double> lu_decompose<double>(
    const linalg::Matrix<double>&, const SolveOptions<double>&);
template CholeskyResult<double> cholesky_decompose<double>(
    const linalg::Matrix<double>&, const SolveOptions<double>&);
template QRResult<double> qr_decompose<double>(
    const linalg::Matrix<double>&, const SolveOptions<double>&);

template SolveResult<double> solve_lu<double>(
    const LUResult<double>&, const linalg::Vector<double>&,
    const SolveOptions<double>&);
template SolveResult<double> solve_cholesky<double>(
    const CholeskyResult<double>&, const linalg::Vector<double>&,
    const SolveOptions<double>&);
template SolveResult<double> solve_qr<double>(
    const QRResult<double>&, const linalg::Vector<double>&,
    const SolveOptions<double>&);

template class Solver<double>;

template double determinant<double>(
    const linalg::Matrix<double>&, const SolveOptions<double>&);
template std::size_t numerical_rank<double>(
    const linalg::Matrix<double>&, const SolveOptions<double>&);
template double residual_norm<double>(
    const linalg::Matrix<double>&,
    const linalg::Vector<double>&,
    const linalg::Vector<double>&);

} // namespace solver
} // namespace math
