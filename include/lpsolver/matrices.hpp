#ifndef MATRICES_HPP
#define MATRICES_HPP

#include <lpsolver/structs.hpp>

namespace LPSolver {
    Matrix select_columns(const Matrix &, const std::vector<int> &);

    Matrix construct_block(const std::vector<std::vector<Matrix>> &);

    Matrix construct_diag(const Vector &);
} // namespace LPSolver

#endif // MATRICES_HPP
