#ifndef MATRICES_HPP
#define MATRICES_HPP

#include <lpsolver/structs.hpp>

namespace LPSolver {
    Matrix select_columns(const Matrix &, const std::vector<int> &);

    Matrix construct_block(const std::vector<std::vector<Matrix>> &);

    Matrix construct_diag(const Vector &);

    std::vector<Eigen::Triplet<double>> to_triplets(const Matrix &);

    void add_row_to_row(std::vector<std::unordered_map<int, double>> &, std::vector<std::unordered_map<int, double>> &, int, int, double);
} // namespace LPSolver

#endif // MATRICES_HPP
