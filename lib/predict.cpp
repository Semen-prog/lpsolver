#include "lpsolver/matrices.hpp"
#ifdef SUPER
#include <Eigen/SuperLUSupport>
#endif
#include <lpsolver/solver.hpp>

namespace LPSolver {
    Delta predictDirection(const Problem &prob, const Position &position) {
        if (position.cnt_free_indices() != 0) {
            Matrix A1 = select_columns(prob.A, position.get_free_indices());
            Matrix A2 = select_columns(prob.A, position.get_remaining_indices());

            Vector x2 = position.get_x_remaining();
            Vector s = position.get_s_remaining();
            Matrix invH = position.constructInvH();

            Matrix N = A2 * invH * A2.transpose();
            Matrix M = construct_block({{A1, N}, {Matrix(A1.transpose().rows(), A1.cols()), A1.transpose()}});

            #ifdef SUPER
            Eigen::SuperLU<Eigen::SparseMatrix<double>> slu;
            #else
            Eigen::SparseLU<Eigen::SparseMatrix<double>> slu;
            #endif

            slu.compute(M);
            assert(slu.info() == Eigen::Success);

            Vector res1 = A2 * x2;
            Vector right_v(res1.rows() + position.cnt_free_indices());
            for (int i = 0; i < res1.rows(); ++i) {
                right_v(i) = res1(i);
            }
            for (int i = res1.rows(); i < right_v.rows(); ++i) {
                right_v(i) = 0;
            }

            Vector sol = slu.solve(right_v);

            Vector delta_x1(position.cnt_free_indices());
            Vector delta_y(sol.rows() - position.cnt_free_indices());
            for (int i = 0; i < position.cnt_free_indices(); ++i) {
                delta_x1(i) = sol(i);
            }
            for (int i = position.cnt_free_indices(); i < sol.rows(); ++i) {
                delta_y(i - position.cnt_free_indices()) = sol(i);
            }

            Vector delta_s = -A2.transpose() * delta_y;

            Vector delta_x2 = -invH * delta_s - x2;

            Vector true_deltax(position.n);
            true_deltax.setZero();
            std::vector<int> free_indices = position.get_free_indices();
            std::vector<int> remaining_indices = position.get_remaining_indices();
            for (size_t i = 0; i < free_indices.size(); ++i) {
                true_deltax(free_indices[i]) = delta_x1(i);
            }
            for (size_t i = 0; i < remaining_indices.size(); ++i) {
                true_deltax(remaining_indices[i]) = delta_x2(i);
            }

            Vector true_deltas(position.n);
            true_deltas.setZero();
            for (size_t i = 0; i < remaining_indices.size(); ++i) {
                true_deltas(remaining_indices[i]) = delta_s(i);
            }

            return Delta(position.n, position.m, true_deltax, delta_y, true_deltas);
        } else {
            Matrix A2 = select_columns(prob.A, position.get_remaining_indices());
            Vector x2 = position.get_x_remaining();
            Vector s = position.get_s_remaining();
            Matrix invH = position.constructInvH();

            #ifdef SUPER
            Eigen::SuperLU<Eigen::SparseMatrix<double>> slu;
            #else
            Eigen::SparseLU<Eigen::SparseMatrix<double>> slu;
            #endif

            Matrix slu_matr = A2 * invH * A2.transpose();
            slu.compute(slu_matr);
            assert(slu.info() == Eigen::Success);

            Vector delta_y = slu.solve(A2 * x2);
            Vector delta_s = -A2.transpose() * delta_y;
            Vector delta_x2 = -invH * delta_s - x2;

            Vector true_deltax(position.n);
            true_deltax.setZero();
            Vector true_deltas(position.n);
            true_deltas.setZero();

            std::vector<int> remaining_indices = position.get_remaining_indices();
            for (size_t i = 0; i < remaining_indices.size(); ++i) {
                true_deltax(remaining_indices[i]) = delta_x2(i);
                true_deltas(remaining_indices[i]) = delta_s(i);
            }

            return Delta(position.n, position.m, true_deltax, delta_y, true_deltas);
        }
    }

    double predictLength(const Position &position, const Delta &delta, double gamma_predict) {
        double step = 1e-3;

        auto ok = [&](double x) {
            return (position + delta.remaining() * x).isCorrect() && (position + delta.remaining() * x).gamma() >= gamma_predict;
        };

        while (!ok(step)) {
            step /= 2;
        }

        while (ok(step * 2)) {
            step *= 2;
        }

        double left = step, right = step * 2;
        while (right - left > 1e-2) {
            double mid = (left + right) / 2;
            if (ok(mid)) left = mid;
            else right = mid;
        }
        return left;
    }
};
