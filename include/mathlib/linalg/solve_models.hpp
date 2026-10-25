
#pragma once

#include "mathlib/linalg/matrix.hpp"
#include "mathlib/linalg/vector.hpp"

#include <cstddef>
#include <vector>

namespace math {
namespace solver {

enum class SolveMethod {
    Automatic,
    GaussianElimination,
    LU,
    Cholesky,
    QR,
    ConjugateGradient
};

enum class SolveStatus {
    Success,
    InvalidDimensions,
    SingularMatrix,
    NotPositiveDefinite,
    NoConvergence,
    NonFiniteInput,
    NumericalFailure,
    UnsupportedMethod
};

template <typename T>
struct SolveResult {
    SolveStatus status;
    linalg::Vector<T> solution;
    double residual_norm;
    std::size_t iterations;

    SolveResult()
        : status(SolveStatus::NumericalFailure),
          solution(0),
          residual_norm(0.0),
          iterations(0) {}

    bool success() const {
        return status == SolveStatus::Success;
    }
};

namespace detail {

template <typename T>
double value_as_double(const T& value);

template <typename T>
linalg::Matrix<double> to_double_matrix(
    const linalg::Matrix<T>& A);

template <typename T>
linalg::Vector<double> to_double_vector(
    const linalg::Vector<T>& b);

template <typename T>
linalg::Vector<T> to_scalar_vector(
    const linalg::Vector<double>& b);

template <typename T>
bool matrix_is_finite(const linalg::Matrix<T>& A);

template <typename T>
bool vector_is_finite(const linalg::Vector<T>& b);

inline double vector_norm(
    const linalg::Vector<double>& v);

inline double dot_double(
    const linalg::Vector<double>& a,
    const linalg::Vector<double>& b);

inline linalg::Vector<double> matrix_vector_product(
    const linalg::Matrix<double>& A,
    const linalg::Vector<double>& x);

template <typename T>
SolveResult<T> make_failure(SolveStatus status);

}

template <typename T>
struct SolveOptions {
    T tolerance;
    T pivot_tolerance;
    std::size_t max_iterations;
    bool check_finite;

    SolveOptions()
        : tolerance(static_cast<T>(1e-6)),
          pivot_tolerance(static_cast<T>(1e-7)),
          max_iterations(100),
          check_finite(true) {}
};

template <typename T>
struct LUResult {
    SolveStatus status;
    linalg::Matrix<T> L;
    linalg::Matrix<T> U;
    std::vector<std::size_t> permutation;
    int permutation_sign;

    LUResult()
        : status(SolveStatus::NumericalFailure),
          L(),
          U(),
          permutation(),
          permutation_sign(1) {}

    bool success() const {
        return status == SolveStatus::Success;
    }
};

template <typename T>
struct CholeskyResult {
    SolveStatus status;
    linalg::Matrix<T> L;

    CholeskyResult()
        : status(SolveStatus::NumericalFailure),
          L() {}

    bool success() const {
        return status == SolveStatus::Success;
    }
};

template <typename T>
struct QRResult {
    SolveStatus status;
    linalg::Matrix<T> Q;
    linalg::Matrix<T> R;
    std::size_t rank;

    QRResult()
        : status(SolveStatus::NumericalFailure),
          Q(),
          R(),
          rank(0) {}

    bool success() const {
        return status == SolveStatus::Success;
    }
};

template <typename T>
LUResult<T> lu_decompose(
    const linalg::Matrix<T>& A,
    const SolveOptions<T>& options = SolveOptions<T>());

template <typename T>
CholeskyResult<T> cholesky_decompose(
    const linalg::Matrix<T>& A,
    const SolveOptions<T>& options = SolveOptions<T>());

template <typename T>
QRResult<T> qr_decompose(
    const linalg::Matrix<T>& A,
    const SolveOptions<T>& options = SolveOptions<T>());

template <typename T>
SolveResult<T> solve_lu(
    const LUResult<T>& decomposition,
    const linalg::Vector<T>& b,
    const SolveOptions<T>& options = SolveOptions<T>());

template <typename T>
SolveResult<T> solve_cholesky(
    const CholeskyResult<T>& decomposition,
    const linalg::Vector<T>& b,
    const SolveOptions<T>& options = SolveOptions<T>());

template <typename T>
SolveResult<T> solve_qr(
    const QRResult<T>& decomposition,
    const linalg::Vector<T>& b,
    const SolveOptions<T>& options = SolveOptions<T>());

template <typename T>
class Solver {
public:
    static SolveResult<T> solve(
        const linalg::Matrix<T>& A,
        const linalg::Vector<T>& b,
        SolveMethod method = SolveMethod::Automatic,
        const SolveOptions<T>& options = SolveOptions<T>());

    // Solve multiple right-hand sides: AX = B.
    static linalg::Matrix<T> solve(
        const linalg::Matrix<T>& A,
        const linalg::Matrix<T>& B,
        SolveMethod method = SolveMethod::Automatic,
        const SolveOptions<T>& options = SolveOptions<T>());
};

template <typename T>
T determinant(
    const linalg::Matrix<T>& A,
    const SolveOptions<T>& options = SolveOptions<T>());

template <typename T>
std::size_t numerical_rank(
    const linalg::Matrix<T>& A,
    const SolveOptions<T>& options = SolveOptions<T>());

template <typename T>
double residual_norm(
    const linalg::Matrix<T>& A,
    const linalg::Vector<T>& x,
    const linalg::Vector<T>& b);

} // namespace solver
} // namespace math
