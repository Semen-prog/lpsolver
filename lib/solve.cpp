#include "lpsolver/matrices.hpp"
#include "lpsolver/structs.hpp"
#include <lpsolver/solver.hpp>
#include <optional>

namespace LPSolver {
    #ifdef SUPER
    using Solver = Eigen::SuperLU<Eigen::SparseMatrix<double>>;
    #else
    using Solver = Eigen::SparseLU<Eigen::SparseMatrix<double>>;
    #endif
    void step(const Problem &prob, Position &position, const Delta &delta, double len) {
        position += delta * len;
        Vector nw_s = position.s;
        std::vector<int> zero_indices = position.get_zero_indices();
        Vector add = select_columns(prob.A.transpose(), zero_indices) * position.y;
        for (size_t i = 0; i < zero_indices.size(); ++i) {
            nw_s(zero_indices[i]) = prob.c(zero_indices[i]) - add(i);
        }
        position.s = nw_s;

        double primal = prob.primal_value(position.x);
        double dual = prob.dual_value(position.y);
        if (dual > primal) {
            debug_print("{0}\n", (prob.A.transpose() * position.y + position.s - prob.c).cwiseAbs().maxCoeff());
            debug_print("{0} {1}\n", dual, primal);
            debug_print("{0}\n", (prob.A * position.x - prob.b).cwiseAbs().maxCoeff());
        }
        debug_print("{0} {1}\n", (prob.A.transpose() * position.y + position.s - prob.c).cwiseAbs().maxCoeff(), (prob.A * position.x - prob.b).cwiseAbs().maxCoeff());

        assert((prob.A.transpose() * position.y + position.s - prob.b).cwiseAbs().maxCoeff() < 1e-6);
        assert((prob.A * position.x - prob.b).cwiseAbs().maxCoeff() < 1e-6);
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

    std::tuple<int, std::optional<Vector>, std::optional<Vector>, std::optional<Vector>, std::optional<double>> solve_ellipsoidal_system(
        const Problem &prob,
        const Position &position,
        const Matrix &A2,
        const std::tuple<Vector, Vector, Matrix> &invQ,
        const std::tuple<Vector, Vector, Matrix> &Q,
        const Vector &c,
        const Vector &c2,
        const Vector &b,
        const std::vector<int> &free,
        const std::vector<int> &remaining,
        EllipsoidalBounds bound,
        int j
    )
    {
        double lambd;
        double lambda_inv;
        bool solvable = true;
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
            Vector coeff_0_vec = std::get<0>(invQ) * (std::get<1>(invQ).dot(c2 - vec)) - std::get<2>(invQ) * (c2 - vec);

            double coeff_0 = coeff_0_vec.dot(std::get<0>(Q) * (std::get<1>(Q).dot(coeff_0_vec)) - std::get<2>(Q) * coeff_0_vec);

            vec = A2.transpose() * solve_sparse_with_one_rank(slu, u, v, b);

            Vector coeff_1_vec = std::get<0>(invQ) * (std::get<1>(invQ) * vec) - (std::get<2>(invQ) * vec);

            double coeff_1 = coeff_1_vec.dot(std::get<0>(Q) * (std::get<1>(Q).dot(coeff_1_vec)) - std::get<2>(Q) * coeff_1_vec);
            
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

            true_y += solve_sparse_with_one_rank(slu, u, v, A2 * (std::get<0>(invQ) * (std::get<1>(invQ).dot(c2)) - std::get<2>(invQ) * c2)) - lambd * b; 

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
            for (size_t i = 0; i < remaining.size(); ++i) {
                true_s(remaining[i]) += s(i);
            }

            double cost;
            if (bound == UPPER) {
                cost = true_x.dot(prob.c);
            } else {
                cost = true_y.dot(prob.b);
            }
            return std::make_tuple(0, true_x, true_y, true_s, cost);
        } else {
            return std::make_tuple(1, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
        }
    }

    decltype(auto) ellipsoidal_bound(const Problem &prob, Position &position, int j, Vector &w, EllipsoidalBounds bound) {
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

            Q = std::make_tuple(w2.cwiseInverse(), w2.cwiseInverse(), construct_diag(w2.cwiseInverse().cwiseProduct(w2.cwiseInverse())));
            invQ = std::make_tuple(w2 * (1 / (n2 - 1)), w2, construct_diag(w2.cwiseProduct(w2)));
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

            auto [status, true_x, true_y, true_s, cost] = solve_ellipsoidal_system_with_free(A1, A2, invQ, Q, c, c1, c2, b, free, remaining, bound, j);

            if (status != 0) {
                // return ((None, None, None, None), 2)
            }

            if (bound == UPPER && (true_b - A * true_x).cwiseAbs().maxCoeff() > ellipsoidal_acc) {
                // return ((None, None, None, None), 1)
            }
            if (bound == LOWER && (true_c - A.transpose() * true_y - true_s).cwiseAbs().maxCoeff() > ellipsoidal_acc) {
                // return ((None, None, None, None), 1)
            }

            Vector x_remaining(remaining.size());
            for (size_t i = 0; i < remaining.size(); ++i) {
                x_remaining(i) = true_x(remaining[i]);
            }
            Vector s_remaining(remaining.size());
            for (size_t i = 0; i < remaining.size(); ++i) {
                s_remaining(i) = true_s(remaining[i]);
            }
            if ((x_remaining.minCoeff() >= -1e-5 && bound == UPPER) ||
                (s_remaining.minCoeff() >= -1e-5 && bound == LOWER)) {
                return std::make_tuple(std::make_tuple(cost, true_x, true_y, true_s), 0);
            } else {
                // return ((None, None, None, None), 1)
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

            auto [status, true_x, true_y, true_s, cost] = solve_ellipsoidal_system(prob, position, A2, invQ, Q, c, c2, b, free, remaining, bound, j);

            if (status != 0) {
                // return ((None, None, None, None), status)
            }

            if (bound == UPPER && (true_b - A * *true_x).cwiseAbs().maxCoeff() > ellipsoidal_acc) {
                // return ((None, None, None, None), 1)
            }

            if (bound == LOWER && (true_c - A.transpose() * *true_y - *true_s).cwiseAbs().maxCoeff() > ellipsoidal_acc) {
                // return ((None, None, None, None), 1)
            }

            Vector true_x_remaining(remaining.size());
            Vector true_s_remaining(remaining.size());
            for (size_t i = 0; i < remaining.size(); ++i) {
                true_x_remaining(i) = (*true_x)(remaining[i]);
                true_s_remaining(i) = (*true_s)(remaining[i]);
            }

            if ((true_x_remaining.minCoeff() >= -1e-5 && bound == UPPER) ||
                (true_s_remaining.minCoeff() >= -1e-5 && bound == LOWER)) {
                return std::make_tuple(std::make_tuple(cost, true_x, true_y, true_s), 0);
            } else {
                // return ((None, None, None, None), 1)
            }
        }
    }

    void filter_variables(const Problem &prob, Position &position) {
        std::vector<int> remaining_indices = position.get_remaining_indices();
        for (int i : remaining_indices) {
            Vector w(position.x.rows());
            w.setZero();
            for (size_t j = 0; j < remaining_indices.size(); ++j) {
                w(remaining_indices[j]) = sqrt(position.x(remaining_indices[j]) / position.s(remaining_indices[j]));
            }
            auto [tuple1, status] = ellipsoidal_bound(prob, position, i, w, UPPER);
            auto [upper_bound, x_u, y_u, s_u] = tuple1;
            
            assert(status != 2);
        }
    }

    Position solve(const Problem &prob, const Position &init, double eps, double gamma_center, double gamma_predict) {
        int cnt_iter = 0;
        debug_print("Called solve\n");
        Position position = init;
        while (position.mu() > eps) {
            ++cnt_iter;
            while (position.gamma() < gamma_center) {
                Delta delta = centralDirection(prob, position);
                double length = centralLength(position, delta);
                // position += delta * length;
                step(prob, position, delta, length);
                if (prob.primal_value(position.x) - prob.dual_value(position.y) < 0.1 && position.n - position.index_zero.size() - position.index_free.size() > 2) {
                    filter_variables(prob, position);
                }
                debug_print("center step: mu = {0}, gamma = {1}\n", position.mu(), position.gamma());
            }
            Delta delta = predictDirection(prob, position);
            double length = predictLength(position, delta, gamma_predict);
            // position += delta * length;
            step(prob, position, delta, length);
            debug_print("predict step: mu = {0}, gamma = {1}\n", position.mu(), position.gamma());
            debug_print("INFO 1: {0}\n", position.x.dot(prob.c) - position.y.dot(prob.b));
            debug_print("INFO 2: {0}\n", position.x.cwiseProduct(position.s).cwiseAbs().maxCoeff());
            debug_print("Ax - b max coeff: {0}\n", (prob.A * position.x - prob.b).cwiseAbs().maxCoeff());
        }
        debug_print("iterations count: {0}\n", cnt_iter);
        return position;
    }
} // namespace LPSolver
