#include <lpsolver/matrices.hpp>

namespace LPSolver {
    Matrix select_columns(const Matrix &matr, const std::vector<int> &columns) {
        std::vector<Eigen::Triplet<double>> triplets;

        for (size_t i = 0; i < columns.size(); ++i) {
            int col = columns[i];
            for (int index = matr.outerIndexPtr()[col]; index < matr.outerIndexPtr()[col + 1]; ++index) {
                int row = matr.innerIndexPtr()[index];
                double val = matr.valuePtr()[index];
                // std::cout << row << ' ' << col << ' ' << val << '\n';
                triplets.emplace_back(row, i, val);
            }
        }

        Matrix res(matr.rows(), columns.size());
        res.setFromTriplets(triplets.begin(), triplets.end());
        return res;
    }

    Matrix construct_block(const std::vector<std::vector<Matrix>> &blocks) {
        std::vector<Eigen::Triplet<double>> triplets;
        int cnt_rows = 0;
        int cnt_cols = 0;
        for (size_t i = 0; i < blocks.size(); ++i) {
            cnt_rows += blocks[i][0].rows();
        }
        for (size_t i = 0; i < blocks[0].size(); ++i) {
            cnt_cols += blocks[0][i].cols();
        }

        Matrix res(cnt_rows, cnt_cols);

        int seen_rows = 0;
        for (size_t i = 0; i < blocks.size(); ++i) {
            int seen_cols = 0;
            for (size_t j = 0; j < blocks[i].size(); ++j) {
                if (j > 0) {
                    if (blocks[i][j].rows() != blocks[i][j - 1].rows()) {
                        throw std::runtime_error("bllocks is not a block matrix\n");
                    }
                }
                for (int col = 0; col < blocks[i][j].outerSize(); ++col) {
                    for (int index = blocks[i][j].outerIndexPtr()[col]; index < blocks[i][j].outerIndexPtr()[col + 1]; ++index) {
                        int row = blocks[i][j].innerIndexPtr()[index];
                        double val = blocks[i][j].valuePtr()[index];
                        triplets.emplace_back(seen_rows + row, seen_cols + col, val);
                    }
                }
                seen_cols += blocks[i][j].cols();
            }
            if (seen_cols != cnt_cols) {
                throw std::runtime_error("blocks is not a block matrix\n");
            }
            seen_rows += blocks[i][0].rows();
        }
        if (seen_rows != cnt_rows) {
            throw std::runtime_error("blocks is not a block matrix\n");
        }
        res.setFromTriplets(triplets.begin(), triplets.end());
        return res;
    }

    Matrix construct_diag(const Vector &vec) {
        Matrix res(vec.rows(), vec.rows());
        std::vector<Eigen::Triplet<double>> triplets;
        for (int i = 0; i < vec.rows(); ++i) {
            triplets.emplace_back(i, i, vec(i));
        }
        res.setFromTriplets(triplets.begin(), triplets.end());
        return res;
    }

    std::vector<Eigen::Triplet<double>> to_triplets(const Matrix &m) {
        std::vector<Eigen::Triplet<double>> res;
        for (int col = 0; col < m.outerSize(); ++col) {
            for (int index = m.outerIndexPtr()[col]; index < m.outerIndexPtr()[col + 1]; ++index) {
                int row = m.innerIndexPtr()[index];
                double val = m.valuePtr()[index];
                res.emplace_back(row, col, val);
            }
        }
        return res;
    }

    void add_row_to_row(std::vector<std::unordered_map<int, double>> &rows, std::vector<std::unordered_map<int, double>> &cols, int row1, int row2, double coeff) {
        for (auto &[col, val] : rows[row2]) {
            rows[row1][col] += val * coeff;
            cols[col][row1] += val * coeff;
        }
    }
} // namespace LPSolver
