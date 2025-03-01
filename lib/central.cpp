#include "lpsolver/structs.hpp"
#ifdef SUPER
#include <Eigen/SuperLUSupport>
#endif
#include <LBFGSpp/LBFGSB.h>
#include <lpsolver/solver.hpp>
#include <lpsolver/matrices.hpp>

namespace LPSolver {
    Delta centralDirection(const Problem &prob, const Position &position) {
        debug_print("Called centralDirection\n");
        if (!position.index_free.empty()) {
            std::vector<int> free_indices = position.get_free_indices();
            std::vector<int> remaining_indices = position.get_remaining_indices();

            Matrix A1 = select_columns(prob.A, free_indices);
            Matrix A2 = select_columns(prob.A, remaining_indices);

            Vector x2 = position.get_x_remaining();
            Vector s = position.get_s_remaining();

            Matrix invH = position.constructInvH();

            Matrix A2T = A2.transpose();

            Matrix N = A2 * invH * A2T;

            Matrix M = construct_block({{A1, N}, {Matrix(A1.transpose().rows(), A1.cols()), A1.transpose()}});

            #ifdef SUPER
            Eigen::SuperLU<Eigen::SparseMatrix<double>> slu;
            #else
            Eigen::SparseLU<Eigen::SparseMatrix<double>> slu;
            #endif

            slu.compute(M);
            assert(slu.info() == Eigen::Success);

            Vector right_v(s.rows() + position.cnt_free_indices());

            Vector res1 = A2 * (x2 - position.mu() * s.cwiseInverse());
            for (int i = 0; i < res1.rows(); ++i) {
                right_v(i) = res1(i);
            }
            for (int i = res1.rows(); i < right_v.rows(); ++i) {
                right_v(i) = 0;
            }

            Vector sol = slu.solve(right_v);

            Vector delta_x1(position.cnt_free_indices());
            for (int i = 0; i < position.cnt_free_indices(); ++i) {
                delta_x1(i) = sol(i);
            }
            Vector delta_y(sol.rows() - position.cnt_free_indices());
            for (int i = position.cnt_free_indices(); i < sol.rows(); ++i) {
                delta_y(i - position.cnt_free_indices()) = sol(i);
            }
            Vector delta_s = -A2T * delta_y;
            Vector delta_x2 = -invH * delta_s - x2 + position.mu() * s.cwiseInverse();

            Vector true_deltax(position.x.rows());
            for (int i = 0; i < free_indices.size(); ++i) {
                true_deltax(free_indices[i]) = delta_x1(i);
            }
            for (int i = 0; i < remaining_indices.size(); ++i) {
                true_deltax(remaining_indices[i]) = delta_x2(i);
            }

            Vector true_deltas(position.s.rows());

            for (int i = 0; i < remaining_indices.size(); ++i) {
                true_deltas(remaining_indices[i]) = delta_s(i);
            }

            return Delta(position.n, position.m, true_deltax, delta_y, true_deltas);
        } else {
            Matrix A2 = select_columns(prob.A, position.get_remaining_indices());
            Vector x2 = position.get_x_remaining();
            Vector s2 = position.get_s_remaining();

            std::vector<int> remaining_indices = position.get_remaining_indices();

            Matrix invH = position.constructInvH();
            Matrix A2T = A2.transpose();

            Vector tmp = -x2 + position.mu() * s2.cwiseInverse();

            #ifdef SUPER
            Eigen::SuperLU<Eigen::SparseMatrix<double>> slu;
            #else
            Eigen::SparseLU<Eigen::SparseMatrix<double>> slu;
            #endif

            debug_print("starting multiplication\n");
            Matrix matrix = A2 * invH * A2T;
            debug_print("starting slu operations\n");
            slu.compute(matrix);
            assert(slu.info() == Eigen::Success);

            debug_print("starting solve\n");

            Vector dy = slu.solve(-A2 * tmp);
            debug_print("solved\n");
            Vector ds = -A2T * dy;
            Vector dx2 = -invH * ds + tmp;

            Vector true_deltax = Vector::Constant(position.n, 0);
            Vector true_deltas = Vector::Constant(position.n, 0);

            for (size_t i = 0; i < remaining_indices.size(); ++i) {
                true_deltax(remaining_indices[i]) = dx2(i);
                true_deltas(remaining_indices[i]) = ds(i);
            }

            return Delta(position.n, position.m, true_deltax, dy, true_deltas);

            // Matrix invH = position.constructInvH();
            // Matrix AT = prob.A.transpose();
            // Vector tmp = -position.x + position.mu() * position.s.cwiseInverse();
            // #ifdef SUPER
            // Eigen::SuperLU<Eigen::SparseMatrix<double>> slu;
            // #else
            // Eigen::SparseLU<Eigen::SparseMatrix<double>> slu;
            // #endif
            // debug_print("starting multiplication\n");
            // auto matrix = prob.A * invH * AT;
            // debug_print("starting slu operations\n");
            // slu.compute(matrix);
            // assert(slu.info() == Eigen::Success);
            
            // debug_print("starting solve\n");
            // Vector dy = slu.solve(-prob.A * tmp);
            // debug_print("solved\n");
            // Vector ds = -AT * dy;
            // Vector dx = -invH * ds + tmp;

            // debug_print("Finished centralDirection\n");
            // return Delta(position.n, position.m, dx, dy, ds);
        }
    }

    class Block {
      private:
        Position position_, delta_;
      public:
        Block(const Position &position, const Delta &delta): position_(position), delta_(delta) {}
        double f(double x) {
            return -(position_.x + delta_.x * x).array().log().sum() - (position_.s + delta_.s * x).array().log().sum();
        }
        double df(double x) {
            return -delta_.x.cwiseProduct((position_.x + delta_.x * x).cwiseInverse()).sum() - delta_.s.cwiseProduct((position_.s + delta_.s * x).cwiseInverse()).sum();
        }
        double operator()(const Vector &x, Vector &grad) {
            grad = Vector::Constant(1, df(x(0)));
            return f(x(0));
        }
    };

    double centralLength(const Position &position, const Delta &delta) {
        debug_print("Called centralLength\n");
        double upper_bound = 1e-3;
        double mu = position.mu();
        auto ok = [&](double x) {
            return (position + delta * x).isCorrect() && (position + delta * x).mu() <= 1.1 * mu;
        };
        assert(ok(0));

        while (!ok(upper_bound)) {
            upper_bound /= 2;
        }

        while (ok(upper_bound * 2)) {
            upper_bound *= 2;
        }
        double left = upper_bound, right = upper_bound * 2;

        while (right - left > 1e-2) {
            double mid = (left + right) / 2;
            if (ok(mid)) left = mid;
            else right = mid;
        }
        upper_bound = left;

        LBFGSpp::LBFGSBParam<double> param;
        param.epsilon = 1e-2;
        param.max_iterations = 1000;

        LBFGSpp::LBFGSBSolver<double> solver(param);
        Block fun(position, delta);

        Vector x = Vector::Constant(1, upper_bound);
        double df;
        solver.minimize(fun, x, df, Vector::Constant(1, 1e-9), Vector::Constant(1, upper_bound));
        debug_print("Finished centralLength\n");
        return x(0);
    }
} // namespace LPSolver
