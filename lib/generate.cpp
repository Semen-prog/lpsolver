#include <lpsolver/generate.hpp>
#include <lpsolver/structs.hpp>
#include <numeric>
#include <random>

#ifdef INFO
#include <iostream>
#include <format>
#endif

namespace LPSolver {
    std::pair<Problem, Position> generateProblem(int m, int n, long long max_non_zero, int random_seed) {
#ifdef INFO
        std::print(std::cerr, "called generateProblem with m == {0}, n == {1}, max_non_zero == {2}, random_seed == {3}\n", m, n, max_non_zero, random_seed);
#endif
        constexpr double kDoublePrecisionEps = 1e-9;
        assert(n > m && "n must be greater than m");
        assert(max_non_zero >= m && "max_non_zero must be >= m");

        std::uniform_real_distribution<double> rng(-5, 5);
        std::mt19937 rnd(random_seed);

        std::vector<int> columns(n);
        std::iota(columns.begin(), columns.end(), 0);
        std::shuffle(columns.begin(), columns.end(), rnd);

        std::vector<std::map<int, double>> matrix_sets(m);
        for (int i = 0; i < m; ++i) {
            double cur = 0;
            while (std::abs(cur) < kDoublePrecisionEps) {
                cur = rng(rnd);
            }
            matrix_sets[i].emplace(columns[i], cur);
            --max_non_zero;
        }

        std::uniform_int_distribution<int> rows_distrib(0, m - 1);
        for (int i = m; i < n; ++i) {
            double cur = 0;
            while (std::abs(cur) < kDoublePrecisionEps) {
                cur = rng(rnd);
            }
            matrix_sets[rows_distrib(rnd)].emplace(columns[i], cur);
        }

        int max_iterations = 3 * m;
        while (max_non_zero >= 0 && max_iterations--) {
#ifdef INFO
            std::print(std::cerr, "max_non_zero == {0}\n", max_non_zero);
#endif
            int row1 = rows_distrib(rnd);
            int row2 = rows_distrib(rnd);
            double coefficient = 0;
            while (std::abs(coefficient) < kDoublePrecisionEps) {
                coefficient = rng(rnd);
            }
            int cnt_non_zero_row_1 = matrix_sets[row1].size();
            auto nw_row_1 = matrix_sets[row1];
            for (auto [index, value] : matrix_sets[row2]) {
                nw_row_1[index] += value * coefficient;
                if (std::abs(nw_row_1[index]) < kDoublePrecisionEps) {
                    nw_row_1.erase(index);
                }
            }
            int nw_non_zero_row_1 = nw_row_1.size();
            max_non_zero -= nw_non_zero_row_1 - cnt_non_zero_row_1;
            matrix_sets[row1] = nw_row_1;
        }

        Matrix A(m, n);
        for (int i = 0; i < m; ++i) {
            for (auto [index, value] : matrix_sets[i]) {
                A.coeffRef(i, index) = value;
            }
        }

        Vector x(n);

        for (int i = 0; i < m; ++i) {
            double cur_value = 0;
            while (cur_value < kDoublePrecisionEps) {
                cur_value = std::abs(rng(rnd));
            }
            x.coeffRef(i) = cur_value;
        }

        Vector b = A * x;

        Matrix AT = A.transpose();
        Vector y(m);
        for (int i = 0; i < m; ++i) {
            double cur_value = 0;
            while (std::abs(cur_value) < kDoublePrecisionEps) {
                cur_value = rng(rnd);
            }
            y.coeffRef(i) = cur_value;
        }

        Vector s(n);
        for (int i = 0; i < n; ++i) {
            double cur_value = 0;
            while (cur_value < kDoublePrecisionEps) {
                cur_value = std::abs(rng(rnd));
            }
            s.coeffRef(i) = cur_value;
        }

        Vector c = s + AT * y;

        return std::make_pair<Problem, Position>(Problem(n, m, A, b, c), Position(n, m, x, y, s));
    }
} // namespace LPSolver
