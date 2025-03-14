#include <lpsolver/generate.hpp>
#include <lpsolver/structs.hpp>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace LPSolver {
    std::pair<Problem, Position> generateProblem(int m, int n, long long max_non_zero, int random_seed) {
        debug_print("called generateProblem with m == {0}, n == {1}, max_non_zero == {2}, random_seed == {3}\n", m, n, max_non_zero, random_seed);
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

        int max_iterations = 10 * m;
        while (max_non_zero >= 0 && max_iterations--) {
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

        debug_print("Finished while\n");

        Matrix A(m, n);
        debug_print("Filling A...\n");
        std::vector<Eigen::Triplet<double>> triplets;
        std::uniform_real_distribution<double> rng2(1e4, 2e4);
        for (int i = 0; i < m; ++i) {
            for (auto [index, val] : matrix_sets[i]) {
                int sign = static_cast<int>(val > 0) * 2 - 1;
                // if (std::abs(val) > 1e6) val = rng2(rnd) * (rnd() % 2 * 2 - 1);
		// if (std::abs(val) > 100) val = sign * (100 + (std::abs(val) - 100) * 0.01);
		if (std::abs(val) > 1) {
		    val = sign * std::log(std::abs(val));
		}
		if (std::abs(val) > 1e-6) triplets.emplace_back(i, index, val);
            }
        }
        A.setFromTriplets(triplets.begin(), triplets.end());
        debug_print("Filled A\n");

        Vector x(n);

        debug_print("Filling x...\n");
        for (int i = 0; i < n; ++i) {
            double cur_value = 0;
            while (cur_value < kDoublePrecisionEps) {
                cur_value = std::abs(rng(rnd));
            }
            x(i) = cur_value + 1;
        }
        debug_print("Filled x\n");

        Vector b = A * x;

        Matrix AT = A.transpose();
        Vector y(m);
        debug_print("Filling y...\n");
        for (int i = 0; i < m; ++i) {
            double cur_value = 0;
            while (std::abs(cur_value) < kDoublePrecisionEps) {
                cur_value = rng(rnd);
            }
            y(i) = cur_value;
        }
        debug_print("Filled y\n");

        Vector s(n);
        debug_print("Filling s...\n");
        for (int i = 0; i < n; ++i) {
            double cur_value = 0;
            while (cur_value < kDoublePrecisionEps) {
                cur_value = std::abs(rng(rnd));
            }
            s(i) = cur_value + 1;
        }
        debug_print("Filled s\n");

        Vector c = s + AT * y;

        return std::make_pair<Problem, Position>(Problem(n, m, A, b, c), Position(n, m, x, y, s));
    }
} // namespace LPSolver
