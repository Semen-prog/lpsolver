#include "lpsolver/matrices.hpp"
#include "lpsolver/structs.hpp"
#include <lpsolver/solver.hpp>
#include <optional>
#include <iostream>

#ifdef SUPER
#include <Eigen/SuperLUSupport>
#endif

namespace LPSolver {
    #ifdef SUPER
    using Solver = Eigen::SuperLU<Eigen::SparseMatrix<double>>;
    #else
    using Solver = Eigen::SparseLU<Eigen::SparseMatrix<double>>;
    #endif
    void step(const Problem &prob, Position &position, const Delta &delta, double len) {
        position += delta * len;
        std::vector<int> zero_indices = position.get_zero_indices();
        if (!zero_indices.empty()) {
            Vector nw_s = position.s;
            Vector add = select_rows(prob.A.transpose(), zero_indices) * position.y;
            for (size_t i = 0; i < zero_indices.size(); ++i) {
                nw_s(zero_indices[i]) = prob.c(zero_indices[i]) - add(i);
            }
            position.s = nw_s;
        }

        double primal = prob.primal_value(position.x);
        double dual = prob.dual_value(position.y);
        debug_print("primal: {0}, dual: {1}\n", primal, dual);
        if (dual > primal) {
            debug_print("{0}\n", (prob.A.transpose() * position.y + position.s - prob.c).cwiseAbs().maxCoeff());
            debug_print("{0} {1}\n", dual, primal);
            debug_print("{0}\n", (prob.A * position.x - prob.b).cwiseAbs().maxCoeff());
        }
        debug_print("{0} {1}\n", (prob.A.transpose() * position.y + position.s - prob.c).cwiseAbs().maxCoeff(), (prob.A * position.x - prob.b).cwiseAbs().maxCoeff());
        debug_print("Ax - b max coeff: {0}\n", (prob.A * position.x - prob.b).cwiseAbs().maxCoeff());

        debug_print("A^Ty + s - c max coeff: {}\n", (prob.A.transpose() * position.y + position.s - prob.c).cwiseAbs().maxCoeff());

        // assert((prob.A.transpose() * position.y + position.s - prob.c).cwiseAbs().maxCoeff() < 1e-6);
        // assert((prob.A * position.x - prob.b).cwiseAbs().maxCoeff() < 1e-6);
        assert(dual < primal);
    }

    Vector solve_sparse_with_one_rank(
        const Solver &slu,
        const Vector &u,
        const Vector &v,
        const Vector &b
    )
    {
        Vector y = slu.solve(b);
        Vector z = slu.solve(u);
        assert(abs(1 - v.dot(z)) > 1e-9);
        double lambd = v.dot(y) / (1 - v.dot(z));

        return y + lambd * z;
    }

    Vector find_kernel_vector(const Matrix &A) {
        // debug_print("find_kernel_vector A rows: {}, cols: {}\n", A.rows(), A.cols());
        // debug_print("find_kernel_vector A rank: {}\n", calculate_rank(A));
        // debug_print("find_kernel_vector A sum: {}\n", A.sum());
        
        std::vector<Eigen::Triplet<double>> triplets = to_triplets(A);

        // TODO: replace std::unordered_map with std::map
        std::vector<std::unordered_map<int, double>> rows(A.cols());
        std::vector<std::unordered_map<int, double>> cols(A.rows());

        for (const auto &triplet : triplets) {
            rows[triplet.col()][triplet.row()] = triplet.value();
            cols[triplet.row()][triplet.col()] = triplet.value();
        }

        int n = A.rows();
        int m = A.cols();
        Vector x(n);
        x.setZero();
        std::unordered_set<int> zeros;
        int row = 0;

        for (int i = 0; i < std::min(n, m); ++i) {
            int j = row;
            double found = (cols[i].find(j) != cols[i].end() ? std::abs(cols[i][j]) : 0);
            for (auto [cur_row, value] : cols[i]) {
                if (cur_row >= row) {
                    if (std::abs(value) > found) {
                        found = std::abs(value);
                        j = cur_row;
                    }
                }
            }
            // debug_print("J: {}\n", j);
            // j += row;
            // debug_print("ROWS SIZE: {}, j: {}\n", rows.size(), j);
            if (rows[j].find(i) == rows[j].end() || std::abs(rows[j][i]) < 1e-3) {
                x(i) = 1;
                zeros.insert(i);
            } else {
                double div = rows[j][i];
                for (auto &[col, val] : rows[j]) {
                    val /= div;
                    cols[col][j] /= div;
                }

                for (auto [col, val] : rows[j]) {
                    cols[col].erase(j);
                }
                for (auto [col, val] : rows[row]) {
                    cols[col].erase(row);
                }

                std::swap(rows[j], rows[row]);

                for (auto [col, val] : rows[j]) {
                    cols[col].emplace(j, val);
                }
                for (auto [col, val] : rows[row]) {
                    cols[col].emplace(row, val);
                }

                for (int k = row + 1; k < m; ++k) {
                    if (rows[k].contains(i) && std::abs(rows[k][i]) > 1e-3) {
                        add_row_to_row(rows, cols, k, row, -rows[k][i]);
                    }
                }
                row += 1;
                // debug_print("ROW: {}\n", row);
            }
        }
        row -= 2;
        for (int i = 2; i < n + 1; ++i) {
            int col = n - i;
            if (!zeros.contains(col)) {
                Vector tail(n - (col + 1));
                tail.setZero();
                Vector x_tail(n - (col + 1));
                // debug_print("2ROWS SIZE: {}, row: {}\n", rows.size(), row);
                for (auto [cur_col, val] : rows[row]) {
                    if (cur_col >= col + 1) {
                        tail(cur_col - (col + 1)) = val;
                    }
                }
                for (int j = col + 1; j < n; ++j) {
                    x_tail(j - (col + 1)) = x(j);
                }
                x(col) = -tail.dot(x_tail);
                row -= 1;
            } else {
                continue;
            }
        }
        if ((A.transpose() * x).cwiseAbs().maxCoeff() >= 1e-6) {
            // std::cerr << x << '\n';
            // std::cerr << (A.transpose() * x).cwiseAbs() << '\n';
            // debug_print_vector(x);
            // debug_print_vector((A.transpose() * x).cwiseAbs());
            assert(false);
        }
        return x;
    }

    std::tuple<int, std::optional<Vector>, std::optional<Vector>, std::optional<Vector>, std::optional<double>> solve_ellipsoidal_system(
        const Problem &prob,
        const Position &position,
        const Matrix &A2,
        const std::tuple<Vector, Vector, Matrix> &invQ,
        const std::tuple<Vector, Vector, Matrix> &Q,
        const Vector &c2,
        const Vector &b,
        const std::vector<int> &remaining,
        EllipsoidalBounds bound,
        int j
    )
    {
        double lambd;
        double lambda_inv;
        bool solvable = true;
        // debug_print("invQ[2]: {}\n", std::get<2>(invQ).toDense().diagonal().cwiseAbs().minCoeff());
        Matrix N = -A2 * std::get<2>(invQ) * A2.transpose();
        Vector u = -(A2 * std::get<0>(invQ));
        Vector v = (A2 * std::get<1>(invQ));
        #ifdef SUPER
        Eigen::SuperLU<Eigen::SparseMatrix<double>> slu;
        #else
        Eigen::SparseLU<Eigen::SparseMatrix<double>> slu;
        #endif
        slu.compute(N);
        if (slu.info() != Eigen::Success) {
            solvable = false;
        }

        if (solvable) {
            Vector vec = A2.transpose() * solve_sparse_with_one_rank(slu, u, v, A2 * (std::get<0>(invQ) * (std::get<1>(invQ).dot(c2)) - std::get<2>(invQ) * c2));
            // debug_print("solve_ellipsoidal_system vec sum: {}\n", vec.sum());
            Vector coeff_0_vec = std::get<0>(invQ) * (std::get<1>(invQ).dot(c2 - vec)) - std::get<2>(invQ) * (c2 - vec);

            double coeff_0 = coeff_0_vec.dot(std::get<0>(Q) * (std::get<1>(Q).dot(coeff_0_vec)) - std::get<2>(Q) * coeff_0_vec);

            // debug_print("coeff_0: {}\n", coeff_0);

            vec = A2.transpose() * solve_sparse_with_one_rank(slu, u, v, b);

            Vector coeff_1_vec = std::get<0>(invQ) * (std::get<1>(invQ).dot(vec)) - (std::get<2>(invQ) * vec);

            double coeff_1 = coeff_1_vec.dot(std::get<0>(Q) * (std::get<1>(Q).dot(coeff_1_vec)) - std::get<2>(Q) * coeff_1_vec);

            // debug_print("coeff_1: {}\n", coeff_1);
            
            if (coeff_0 / coeff_1 > 0) {
                return std::make_tuple(1, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
            }

            lambd = sqrt(-coeff_0 / coeff_1);
            lambda_inv = 1 / lambd;
        } else {
            return std::make_tuple(2, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
        }

        if (solvable && lambda_inv < 1e6) {
            Vector true_x(position.x.rows());
            Vector true_y(position.y.rows());
            Vector true_s(position.s.rows());

            true_x.setZero();
            true_y.setZero();
            true_s.setZero();

            true_y += solve_sparse_with_one_rank(slu, u, v, A2 * (std::get<0>(invQ) * (std::get<1>(invQ).dot(c2)) - std::get<2>(invQ) * c2) - lambd * b); 
            // debug_print_vector(true_y);

            Vector vec = (
                lambda_inv * c2
                -A2.transpose()
                * (
                    solve_sparse_with_one_rank(slu, u, v, 
                        A2
                        * (
                            lambda_inv * std::get<0>(invQ) * std::get<1>(invQ).dot(c2)
                            - lambda_inv * std::get<2>(invQ) * c2
                        )
                    )
                )
            ) + A2.transpose() * (solve_sparse_with_one_rank(slu, u, v, b));

            Vector x2 = std::get<0>(invQ) * std::get<1>(invQ).dot(vec) - std::get<2>(invQ) * vec;

            Vector s = (
                c2 - A2.transpose() * (solve_sparse_with_one_rank(slu, u, v, A2 * (std::get<0>(invQ) * std::get<1>(invQ).dot(c2) - std::get<2>(invQ) * c2)))
            ) + A2.transpose() * (solve_sparse_with_one_rank(slu, u, v, lambd * b));

            for (size_t i = 0; i < remaining.size(); ++i) {
                true_x(remaining[i]) += x2(i);
            }

            std::vector<int> free_s(position.index_zero.begin(), position.index_zero.end());
            if (bound == LOWER) {
                free_s.emplace_back(j);
            }
            std::sort(free_s.begin(), free_s.end());

            if (!free_s.empty()) {
                Vector sub = select_columns(prob.A, free_s).transpose() * true_y;
                for (size_t i = 0; i < free_s.size(); ++i) {
                    true_s(free_s[i]) += (
                        prob.c(free_s[i]) - sub(i)
                    );
                }
            }
            // debug_print_vector(true_s);
            // debug_print_vector(s);
            for (size_t i = 0; i < remaining.size(); ++i) {
                true_s(remaining[i]) += s(i);
            }

            double cost;
            if (bound == UPPER) {
                cost = true_x.dot(prob.c);
            } else {
                cost = true_y.dot(prob.b);
            }

            // debug_print("{} {}\n{} {}\n", true_x.sum(), prob.c.sum(), true_y.sum(), prob.b.sum());

            // debug_print("!!!!!!!!!!!!! {}\n", (A2.transpose() * position.y + s - c2).cwiseAbs().maxCoeff());

            // assert((A2 * x2 - b).cwiseAbs().maxCoeff() < 1e-3);
            // assert((A2.transpose() * position.y + s - c2).cwiseAbs().maxCoeff() < 1e-3);

            // debug_print_vector(true_s);
            
            return std::make_tuple(0, true_x, true_y, true_s, cost);
        } else {
            return std::make_tuple(1, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
        }
    }

    std::tuple<int, std::optional<Vector>, std::optional<Vector>, std::optional<Vector>, std::optional<double>> solve_ellipsoidal_system_with_free(
        const Problem &prob,
        const Position &position,
        const Matrix &A1,
        const Matrix &A2,
        const std::tuple<Vector, Vector, Matrix> &invQ,
        const std::tuple<Vector, Vector, Matrix> &Q,
        const Vector &c,
        const Vector &c1,
        const Vector &c2,
        const Vector &b,
        const std::vector<int> &free,
        const std::vector<int> &remaining,
        EllipsoidalBounds bound,
        int j
    )
    {
        /*
        debug_print("free:");
        for (int el : free) {
            debug_print(" {}", el);
        }
        debug_print("\nremaining:");
        for (int el : remaining) {
            debug_print(" {}", el);
        }
        debug_print("\n");
        debug_print("position x sum: {}\n", position.x.sum());
        */
        int n_free = A1.cols();
        Vector A2iq0 = A2 * std::get<0>(invQ);
        Vector u(A2iq0.rows() + n_free);
        u.setZero();
        for (int i = 0; i < A2iq0.rows(); ++i) {
            u(i) = A2iq0(i);
        }

        Vector A2iq1 = A2 * std::get<1>(invQ);
        Vector v(n_free + A2iq1.rows());
        v.setZero();
        for (int i = 0; i < A2iq1.rows(); ++i) {
            v(n_free + i) = A2iq1(i);
        }

        Matrix M = construct_block({{A1, A2 * std::get<2>(invQ) * A2.transpose()}, {Matrix(A1.transpose().rows(), A1.cols()), A1.transpose()}});

        Solver slu;
        slu.compute(M);
        if (slu.info() != Eigen::Success) {
            return std::make_tuple(2, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
        }
        
        Vector right_b_lambda(b.rows() + c1.rows());
        right_b_lambda.setZero();
        for (int i = 0; i < b.rows(); ++i) {
            right_b_lambda(i) = b(i);
        }
        Vector sol_lambda = solve_sparse_with_one_rank(slu, u, v, right_b_lambda);
        
        Vector right_b_free_top = -A2 * (std::get<0>(invQ) * std::get<1>(invQ).dot(c2) - std::get<2>(invQ) * c2);
        Vector right_b_free(right_b_free_top.rows() + c1.rows());
        for (int i = 0; i < right_b_free_top.rows(); ++i) {
            right_b_free(i) = right_b_free_top(i);
        }
        for (int i = 0; i < c1.rows(); ++i) {
            right_b_free(i + right_b_free_top.rows()) = c1(i);
        }
        Vector sol_free = solve_sparse_with_one_rank(slu, u, v, right_b_free);

        Vector y_lambda(sol_lambda.rows() - n_free);
        for (int i = n_free; i < sol_lambda.rows(); ++i) {
            y_lambda(i - n_free) = sol_lambda(i);
        }
        Vector y_free(sol_free.rows() - n_free);
        for (int i = n_free; i < sol_free.rows(); ++i) {
            y_free(i - n_free) = sol_free(i);
        }

        Vector x2_lambda = -A2.transpose() * y_lambda;
        Vector x2_free = c2 - A2.transpose() * y_free;

        double coeff_free = (
            x2_free.cwiseProduct(std::get<0>(invQ) * std::get<1>(invQ).dot(x2_free) - std::get<2>(invQ) * x2_free)
        ).sum();
        double coeff_sq = (
            x2_lambda.cwiseProduct(std::get<0>(invQ) * std::get<1>(invQ).dot(x2_lambda) - std::get<2>(invQ) * x2_lambda)
        ).sum();

        if (std::abs(coeff_sq) > 1e-12 && coeff_free / coeff_sq > 0) {
            return std::make_tuple(1, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
        }

        // debug_print("coeff_free: {}, coeff_sq: {}\n", coeff_free, coeff_sq);

        double lambd = sqrt(-coeff_free / coeff_sq);
        double lambda_inv = 1 / lambd;
        if (lambda_inv > 1e6) {
            return std::make_tuple(1, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
        }

        Vector y = lambd * y_lambda + y_free;
        Vector vec = c2 - A2.transpose() * y;
        // debug_print("vec sum: {}\n", vec.sum());
        // debug_print("lambd: {}, invq[0] sum: {}, invq[1] sum: {}, invq[2] sum: {}\n", lambd, std::get<0>(invQ).sum(), std::get<1>(invQ).sum(), std::get<2>(invQ).sum());
        Vector x2 = lambda_inv * (std::get<0>(invQ) * std::get<1>(invQ).dot(vec) - std::get<2>(invQ) * vec);
        Vector sol_lambda_top(n_free);
        Vector sol_free_top(n_free);
        for (int i = 0; i < n_free; ++i) {
            sol_lambda_top(i) = sol_lambda(i);
            sol_free_top(i) = sol_free(i);
        }
        Vector x1 = (
            sol_lambda_top + lambda_inv * sol_free_top
        );

        Vector true_x(position.x.rows());
        true_x.setZero();

        Vector true_y(position.y.rows());
        true_y.setZero();

        Vector true_s(position.s);
        true_s.setZero();

        for (size_t i = 0; i < free.size(); ++i) {
            true_x(free[i]) += x1(i);
        }
        for (size_t i = 0; i < remaining.size(); ++i) {
            true_x(remaining[i]) += x2(i);
        }

        Vector s = lambd * (std::get<0>(Q) * std::get<1>(Q).dot(x2) - std::get<2>(Q) * x2);

        true_y = y;

        double cost;

        if (bound == UPPER) {
            cost = true_x.dot(prob.c);
        } else {
            cost = true_y.dot(prob.b);
        }
        
        std::vector<int> free_s(position.index_zero.begin(), position.index_zero.end());
        if (bound == LOWER) {
            free_s.emplace_back(j);
        }
        std::sort(free_s.begin(), free_s.end());

        if (!free_s.empty()) {
            Vector sub = select_columns(prob.A, free_s).transpose() * true_y;
            for (size_t i = 0; i < free_s.size(); ++i) {
                true_s(free_s[i]) += c(free_s[i]) - sub(i);
            }
        }
        for (size_t i = 0; i < remaining.size(); ++i) {
            true_s(remaining[i]) += s(i);
        }

        Vector c_remaining(remaining.size());
        for (int i = 0; i < (int)remaining.size(); ++i) {
            c_remaining(i) = c(remaining[i]);
        }

        // debug_print("true_x sum: {}, true_y sum: {}, true_s sum {}\n", true_x.sum(), true_y.sum(), true_s.sum());
        // debug_print("x1 sum: {}, x2 sum: {}\n", x1.sum(), x2.sum());

        // debug_print("(A1 * x1 + A2 * x2 - b).cwiseAbs().maxCoeff(): {}\n", (A1 * x1 + A2 * x2 - b).cwiseAbs().maxCoeff());
        if ((A1 * x1 + A2 * x2 - b).cwiseAbs().maxCoeff() >= 1e-3) {
            std::cout << (A1 * x1 + A2 * x2 - b).cwiseAbs().maxCoeff() << '\n';
        }
        // assert((A1 * x1 + A2 * x2 - b).cwiseAbs().maxCoeff() < 1e-3);
        // assert((A2.transpose() * y + s - c_remaining).cwiseAbs().maxCoeff() < 1e-3);

        return std::make_tuple(0, true_x, true_y, true_s, cost);
    }

    std::tuple<std::optional<std::tuple<double, Vector, Vector, Vector>>, int> ellipsoidal_bound(const Problem &prob, Position &position, int j, Vector &w, EllipsoidalBounds bound) {
        // debug_print("[ ELLIPSOIDAL_BOUND ] {0}\n", static_cast<int>(bound));
        std::tuple<Vector, Vector, Matrix> Q;
        std::tuple<Vector, Vector, Matrix> invQ;
        std::vector<int> remaining;
        std::vector<int> free;
        if (bound == UPPER) {
            remaining = position.get_remaining_indices();
            if (std::find(remaining.begin(), remaining.end(), j) != remaining.end()) {
                remaining.erase(std::find(remaining.begin(), remaining.end(), j));
            }
            free = position.get_free_indices();
            free.emplace_back(j);
            std::sort(free.begin(), free.end());
            // debug_print("[ W_INFO ]: {}\n", w.sum());
            Vector w2(remaining.size());
            for (size_t i = 0; i < remaining.size(); ++i) {
                w2(i) = w(remaining[i]);
            }
            int n2 = remaining.size();
            Q = std::make_tuple((w2 * (n2 - 1)).cwiseInverse(), w2.cwiseInverse(), construct_diag(w2.cwiseInverse().cwiseProduct(w2.cwiseInverse())));
            invQ = std::make_tuple(w2, w2, construct_diag(w2.cwiseProduct(w2)));
        } else {
            remaining = position.get_remaining_indices();
            if (std::find(remaining.begin(), remaining.end(), j) != remaining.end()) {
                remaining.erase(std::find(remaining.begin(), remaining.end(), j));
            }
            free = position.get_free_indices();
            Vector w2(remaining.size());
            int n2 = remaining.size();
            for (size_t i = 0; i < remaining.size(); ++i) {
                w2(i) = w(remaining[i]);
            }

            // debug_print("[ W2 ] {0}, {1}\n", w2.sum(), n2);

            Q = std::make_tuple(w2.cwiseInverse(), w2.cwiseInverse(), construct_diag(w2.cwiseInverse().cwiseProduct(w2.cwiseInverse())));
            invQ = std::make_tuple(w2 * (1. / (n2 - 1)), w2, construct_diag(w2.cwiseProduct(w2)));
        }

        Matrix A2 = select_columns(prob.A, remaining);
        if (!free.empty()) {
            Matrix A1 = select_columns(prob.A, free);
            Matrix A = prob.A;
            Vector c = prob.c;
            Vector c1(free.size());
            Vector c2(remaining.size());
            for (size_t i = 0; i < free.size(); ++i) {
                c1(i) = c(free[i]);
            }
            for (size_t i = 0; i < remaining.size(); ++i) {
                c2(i) = c(remaining[i]);
            }
            Vector b = prob.b;
            Vector true_b = prob.b;
            Vector true_c = c;

            // debug_print("[ !LOG ]: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n", j, A1.sum(), A2.sum(), std::get<0>(invQ).sum(), std::get<1>(invQ).sum(), std::get<2>(invQ).sum(), std::get<0>(Q).sum(), std::get<1>(Q).sum(), std::get<2>(Q).sum(), c.sum(), c1.sum(), c2.sum(), b.sum(), static_cast<int>(bound));
            // debug_print("index_free:");
            // for (auto el : position.index_free) {
            //     debug_print(" {}", el);
            // }
            // debug_print("\nindex_zero:");
            // for (auto el : position.index_zero) {
            //     debug_print(" {}", el);
            // }
            // debug_print("\n");
            // debug_print("[ LOG ]: calling solve_ellipsoidal_system_with_free\n");

            auto [status, true_x, true_y, true_s, cost] = solve_ellipsoidal_system_with_free(prob, position, A1, A2, invQ, Q, c, c1, c2, b, free, remaining, bound, j);

            // debug_print("[ LOG ] status: {}", status);
            if (status == 0) {
                // debug_print(", true_x sum: {}, true_y sum: {}, true_s sum: {}, cost: {}\n", true_x->sum(), true_y->sum(), true_s->sum(), *cost);
            } else {
                // debug_print("\n");
            }

            if (status != 0) {
                return std::make_tuple(std::nullopt, 2);
            }

            if (bound == UPPER && (true_b - A * *true_x).cwiseAbs().maxCoeff() > ellipsoidal_acc) {
                return std::make_tuple(std::nullopt, 1);
            }
            if (bound == LOWER && (true_c - A.transpose() * *true_y - *true_s).cwiseAbs().maxCoeff() > ellipsoidal_acc) {
                return std::make_tuple(std::nullopt, 1);
            }

            Vector x_remaining(remaining.size());
            for (size_t i = 0; i < remaining.size(); ++i) {
                x_remaining(i) = (*true_x)(remaining[i]);
            }
            Vector s_remaining(remaining.size());
            for (size_t i = 0; i < remaining.size(); ++i) {
                s_remaining(i) = (*true_s)(remaining[i]);
            }
            if ((x_remaining.minCoeff() >= -1e-5 && bound == UPPER) ||
                (s_remaining.minCoeff() >= -1e-5 && bound == LOWER)) {
                return std::make_tuple(std::make_tuple(*cost, *true_x, *true_y, *true_s), 0);
            } else {
                return std::make_tuple(std::nullopt, 1);
            }
        } else {
            Matrix A = prob.A;
            Vector c = prob.c;
            Vector c2(remaining.size());
            for (size_t i = 0; i < remaining.size(); ++i) {
                c2(i) = c(remaining[i]);
            }
            Vector b = prob.b;

            Vector true_b = b;
            Vector true_c = c;

            // debug_print("[ LOG ]: {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}\n", A2.sum(), std::get<0>(invQ).sum(), std::get<1>(invQ).sum(), std::get<2>(invQ).sum(), std::get<0>(Q).sum(), std::get<1>(Q).sum(), std::get<2>(Q).sum(), c.sum(), c2.sum(), b.sum(), static_cast<int>(bound));

            auto [status, true_x, true_y, true_s, cost] = solve_ellipsoidal_system(prob, position, A2, invQ, Q, c2, b, remaining, bound, j);

            if (status != 0) {
                return std::make_tuple(std::nullopt, status);
            }

            if (bound == UPPER && (true_b - A * *true_x).cwiseAbs().maxCoeff() > ellipsoidal_acc) {
                return std::make_tuple(std::nullopt, 1);
            }

            if (bound == LOWER && (true_c - A.transpose() * *true_y - *true_s).cwiseAbs().maxCoeff() > ellipsoidal_acc) {
                return std::make_tuple(std::nullopt, 1);
            }

            Vector true_x_remaining(remaining.size());
            Vector true_s_remaining(remaining.size());
            for (size_t i = 0; i < remaining.size(); ++i) {
                true_x_remaining(i) = (*true_x)(remaining[i]);
                true_s_remaining(i) = (*true_s)(remaining[i]);
            }

            if ((true_x_remaining.minCoeff() >= -1e-5 && bound == UPPER) ||
                (true_s_remaining.minCoeff() >= -1e-5 && bound == LOWER)) {
                return std::make_tuple(std::make_tuple(*cost, *true_x, *true_y, *true_s), 0);
            } else {
                return std::make_tuple(std::nullopt, 1);
            }
        }
    }

    void filter_variables(const Problem &prob, Position &position) {
        std::vector<int> remaining_indices_init = position.get_remaining_indices();
        for (int i : remaining_indices_init) {
            std::vector<int> remaining_indices = position.get_remaining_indices();
            Vector w(position.x.rows());
            w.setZero();
            for (size_t j = 0; j < remaining_indices.size(); ++j) {
                w(remaining_indices[j]) = sqrt(position.x(remaining_indices[j]) / position.s(remaining_indices[j]));
            }
            auto [tuple1, status1] = ellipsoidal_bound(prob, position, i, w, UPPER);
            // debug_print("[ LOG ] ellipsoidal_bound returned status {}\n", status1);
            assert(status1 != 2);
            if (status1 == 0) {
                auto [upper_bound, x_u, y_u, s_u] = *tuple1;
                // debug_print("[ LOG ] upper_bound {}, x_u {}, y_u {}, s_u {}\n", upper_bound, x_u.sum(), y_u.sum(), s_u.sum());
                // debug_print("[ LOG ] some info: {0}\n", upper_bound);
                if (upper_bound < prob.dual_value(position.y)) {
                    if (x_u(i) < 0) {
                        double alpha = -x_u(i) / (position.x(i) - x_u(i));
                        position.x = alpha * position.x + (1 - alpha) * x_u;
                        position.index_zero.insert(i);
                        // debug_print("[ LOG ] index zero additional info: {0} {1} ", i, alpha);
                        // debug_print_vector(position.x);
                        // debug_print("\n");
                        continue;
                    }
                }
            }

            remaining_indices = position.get_remaining_indices();
            w.setZero();
            for (size_t j = 0; j < remaining_indices.size(); ++j) {
                w(remaining_indices[j])  = sqrt(position.x(remaining_indices[j]) / position.s(remaining_indices[j]));
            }

            auto [tuple2, status2] = ellipsoidal_bound(prob, position, i, w, LOWER);
            if (status2 == 2) {
                std::vector<int> nonzero;
                for (size_t j = 0; j < prob.n; ++j) {
                    if (!position.index_zero.contains(j) && j != (size_t)i) {
                        nonzero.emplace_back(j);
                    }
                }
                debug_print("Filtered number: {}\n", position.index_free.size() + position.index_zero.size());
                Vector v = find_kernel_vector(select_columns(prob.A, nonzero));
                if (v.cwiseAbs().maxCoeff() > 0) {
                    position.index_free.insert(i);
                    position.y += (position.s(i) / (v.dot(Vector(select_columns(prob.A, {i}))))) * v;
                    position.s(i) = 0;
                } else {
                    assert(false);
                }
                continue;
            }

            if (status2 == 0) {
                auto [lower_bound, x_l, y_l, s_l] = *tuple2;
                if (lower_bound > prob.primal_value(position.x)) {
                    if (s_l(i) < 0) {
                        double alpha = -s_l(i) / (position.s(i) - s_l(i));
                        position.s = alpha * position.s + (1 - alpha) * s_l;
                        position.y = alpha * position.y + (1 - alpha) * y_l;
                        position.index_free.insert(i);
                    }
                }
            }
        }
        debug_print("Filtered\n");
    }

    Position solve(const Problem &prob, const Position &init, double eps, double gamma_center, double gamma_predict) {
        int cnt_iter = 0;
        // debug_print("Called solve\n");
        Position position = init;
        while (position.mu() > eps) {
            ++cnt_iter;
            while (position.gamma() < gamma_center) {
                Delta delta = centralDirection(prob, position);
                double length = centralLength(position, delta);
                // position += delta * length;
                step(prob, position, delta, length);
                if (prob.primal_value(position.x) - prob.dual_value(position.y) < 100 && position.n - position.index_zero.size() - position.index_free.size() > 2) {
                    debug_print("Before filter: {}\n", position.x.dot(position.s));
                    filter_variables(prob, position);
                    debug_print("After filter: {}\n", position.x.dot(position.s));
                }
                debug_print("center step: mu = {0}, gamma = {1}\n", position.mu(), position.gamma());
                debug_print("Free x:");
                for (int el : position.get_free_indices()) {
                    debug_print(" {0}", el);
                }
                debug_print("\n");
                debug_print("Zero x:");
                for (int el : position.get_zero_indices()) {
                    debug_print(" {0}", el);
                }
                debug_print("\n");

                std::cout << "n == " << prob.n << ", free size: " << position.index_free.size() << ", zero size: " << position.index_zero.size() << '\n';
                std::cout << "free%: " << position.index_free.size() * 100.0 / prob.m << '\n';
                std::cout << "zero%: " << position.index_zero.size() * 100.0 / (prob.n - prob.m) << '\n';
            }
            Delta delta = predictDirection(prob, position);
            double length = predictLength(position, delta, gamma_predict);
            // position += delta * length;
            step(prob, position, delta, length);
            debug_print("predict step: mu = {0}, gamma = {1}\n", position.mu(), position.gamma());
            debug_print("INFO 1: {0}\n", position.x.dot(prob.c) - position.y.dot(prob.b));
            debug_print("INFO 2: {0}\n", position.x.cwiseProduct(position.s).cwiseAbs().maxCoeff());
            debug_print("Ax - b max coeff: {0}\n", (prob.A * position.x - prob.b).cwiseAbs().maxCoeff());
            std::cout << "n == " << prob.n << ", free size: " << position.index_free.size() << ", zero size: " << position.index_zero.size() << '\n';
            std::cout << "free%: " << position.index_free.size() * 100.0 / prob.m << '\n';
            std::cout << "zero%: " << position.index_zero.size() * 100.0 / (prob.n - prob.m) << '\n';
            if (position.index_zero.size() == 196) {
                break;
            }
        }
        debug_print("iterations count: {0}\n", cnt_iter);
        return position;
    }
} // namespace LPSolver
