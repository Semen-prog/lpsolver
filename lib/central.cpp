#include <Eigen/SuperLUSupport>
#include <LBFGSpp/LBFGSB.h>
#include <lpsolver/solver.hpp>

namespace LPSolver {
    Delta centralDirection(const Problem &prob, const Position &position) {
        Matrix invH = position.constructInvH();
        Matrix AT = prob.A.transpose();
        Vector tmp = -position.x + position.mu() * position.s.cwiseInverse();

        Eigen::SuperLU<Eigen::SparseMatrix<double>> slu;
        slu.compute(prob.A * invH * AT);
        assert(slu.info() == Eigen::Success);
        
        Vector dy = slu.solve(-prob.A * tmp);
        Vector ds = -AT * dy;
        Vector dx = -invH * ds + tmp;

        return Delta(position.n, position.m, dx, dy, ds);
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
            return -delta_.x.cwiseProduct((position_.x + delta_.x * x).cwiseInverse()).sum() - delta_.s.cwiseProduct((position_.s + delta_.s * x).cwiseInverse()).sum();;
        }
        double operator()(const Vector &x, Vector &grad) {
            grad = Vector::Constant(1, df(x(0)));
            return f(x(0));
        }
    };

    double centralLength(const Position &position, const Delta &delta) {
        double upper_bound = 1e-3;
        double mu = position.mu();
        auto ok = [&](double x) {
            return (position + delta * x).isCorrect() && (position + delta * x).mu() <= 1.1 * mu;
        };

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
        param.epsilon = 1e-6;
        param.max_iterations = 1000;

        LBFGSpp::LBFGSBSolver<double> solver(param);
        Block fun(position, delta);

        Vector x = Vector::Constant(1, upper_bound);
        double df;
        solver.minimize(fun, x, df, Vector::Constant(1, 1e-9), Vector::Constant(1, upper_bound));
        return x(0);
    }
} // namespace LPSolver