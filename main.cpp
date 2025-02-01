#include <LBFGSB.h>
#include <cassert>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SuperLUSupport>
#include <LBFGS.h>
#include <vector>
#include <iostream>

namespace LPSolver {

    using Matrix = Eigen::SparseMatrix<double>;
    using Vector = Eigen::VectorXd;

    struct Problem {
        size_t n, m;
        Matrix A;
        Vector b, c;

        Problem(size_t n_, size_t m_, const Matrix &A_, const Vector &b_, const Vector &c_)
            : n(n_)
            , m(m_)
            , A(A_)
            , b(b_)
            , c(c_)
            {}
    };

    struct Position {
        size_t n, m;
        Vector x, y, s;

        Position(size_t n_, size_t m_, const Vector &x_, const Vector &y_, const Vector &s_)
            : n(n_)
            , m(m_)
            , x(x_)
            , y(y_)
            , s(s_)
            {}

        double mu() const {
            return x.dot(s) / n;
        }

        double gamma() const {
            return x.cwiseProduct(s).minCoeff() / mu();
        }

        bool isCorrect() const {
            return std::min(x.minCoeff(), s.minCoeff()) > 1e-6;
        }

        Matrix constructInvH() const {
            std::vector<Eigen::Triplet<double>> data;
            for (size_t i = 0; i < n; ++i) {
                data.emplace_back(i, i, x[i] / s[i]);
            }
            Matrix invH(n, n);
            invH.setFromTriplets(data.begin(), data.end());
            return invH;
        }

        Position& operator*=(double val) {
            x *= val;
            y *= val;
            s *= val;
            return *this;
        }

        Position& operator/=(double val) {
            return *this *= (1 / val);
        }

        Position operator/(double val) const {
            return *this * (1 / val);
        }

        Position operator*(double val) const {
            Position copy = *this;
            copy *= val;
            return copy;
        }

        Position& operator+=(const Position &other) {
            x += other.x;
            y += other.y;
            s += other.s;
            return *this;
        }

        Position operator+(const Position &other) const {
            Position copy = *this;
            copy += other;
            return copy;
        }

        Position operator-() const {
            return *this * -1;
        }

        Position& operator-=(const Position &other) {
            x -= other.x;
            y -= other.y;
            s -= other.s;
            return *this;
        }

        Position operator-(const Position &other) const {
            Position copy = *this;
            copy -= other;
            return copy;
        }
    };

    using Delta = Position;

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

    Delta predictDirection(const Problem &prob, const Position &position) {
        Matrix invH = position.constructInvH();
        Matrix AT = prob.A.transpose();

        Eigen::SuperLU<Eigen::SparseMatrix<double>> slu;
        slu.compute(prob.A * invH * AT);
        assert(slu.info() == Eigen::Success);

        Vector dy = slu.solve(-prob.A * -position.x);
        Vector ds = -AT * dy;
        Vector dx = -invH * ds - position.x;

        return Delta(position.n, position.m, dx, dy, ds);
    }

    double predictLength(const Position &position, const Delta &delta, double gamma_predict) {
        double step = 1e-3;

        auto ok = [&](double x) {
            return (position + delta * x).isCorrect() && (position + delta * x).gamma() >= gamma_predict;
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

    Position solve(const Problem &prob, const Position &init, double eps, double gamma_center=0.9, double gamma_predict=0.7) {
        Position position = init;
        while (position.mu() > eps) {
            while (position.gamma() < gamma_center) {
                Delta delta = centralDirection(prob, position);
                double length = centralLength(position, delta);
                position += delta * length;
#ifdef INFO
                std::cerr << "solve step: mu = " << position.mu() << ", gamma = " << position.gamma() << std::endl;
#endif
            }
            Delta delta = predictDirection(prob, position);
            double length = predictLength(position, delta, gamma_predict);
            position += delta * length;
#ifdef INFO
            std::cerr << "predict step: mu = " << position.mu() << ", gamma = " << position.gamma() << std::endl;
#endif
        }
        return position;
    }
};

int main() {
    std::vector<Eigen::Triplet<double>> data;
    size_t n, m; std::cin >> n >> m;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double val; std::cin >> val;
            if (std::abs(val) > 1e-6) data.emplace_back(i, j, val);
        }
    }
    LPSolver::Matrix A(m, n);
    A.setFromTriplets(data.begin(), data.end());
    LPSolver::Vector b(m), c(n), x(n), y(m), s(n);
    for (size_t i = 0; i < m; ++i) {
        double val; std::cin >> val;
        b(i) = val;
    }
    for (size_t i = 0; i < n; ++i) {
        double val; std::cin >> val;
        c(i) = val;
    }
    for (size_t i = 0; i < n; ++i) {
        double val; std::cin >> val;
        x(i) = val;
    }
    for (size_t i = 0; i < m; ++i) {
        double val; std::cin >> val;
        y(i) = val;
    }
    for (size_t i = 0; i < n; ++i) {
        double val; std::cin >> val;
        s(i) = val;
    }
    LPSolver::Problem prob(n, m, A, b, c);
    LPSolver::Position position(n, m, x, y, s);
    auto res = LPSolver::solve(prob, position, 1e-2);
    for (size_t i = 0; i < n; ++i) {
        std::cout << res.x(i) << ' ';
    }
    std::cout << '\n';
    for (size_t i = 0; i < m; ++i) {
        std::cout << res.y(i) << ' ';
    }
    std::cout << '\n';
    for (size_t i = 0; i < n; ++i) {
        std::cout << res.s(i) << ' ';
    }
    std::cout << '\n';
}