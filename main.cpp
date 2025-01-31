#include <cassert>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SuperLUSupport>
#include <functional>
#include <vector>

namespace ODGD {
    double minimize(double start, double left, double right, std::function<double(double)> df, double learningRate=0.1, size_t maxIter=1000, double eps=1e-6) {
        double x = start;
        for (size_t i = 0; i < maxIter; ++i) {
            if (x < left - eps) return left;
            if (x > right + eps) return right;
            double grad = df(x);
            x -= learningRate * df(x);
            if (std::abs(grad) < eps) return x;
        }
        return x;
    }
};

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
            Matrix invH;
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
            return *this;
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
            return *this;
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
            int mid = (left + right) / 2;
            if (ok(mid)) left = mid;
            else right = mid;
        }
        upper_bound = left;

        auto df = [&](double x) {
            return -delta.x.cwiseProduct((position.x + delta.x * x).cwiseInverse()).sum() - delta.s.cwiseProduct((position.s + delta.s * x).cwiseInverse()).sum();
        };
        return ODGD::minimize(upper_bound, 1e-5, upper_bound, df);
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
            int mid = (left + right) / 2;
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
            }
            Delta delta = predictDirection(prob, position);
            double length = predictLength(position, delta, gamma_predict);
            position += delta * length;
        }
        return position;
    }
};

int main() {
    
}