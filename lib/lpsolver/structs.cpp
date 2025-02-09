#include <lpsolver/structs.hpp>

#ifdef INFO
void _printVector(const Eigen::VectorXd& vec) {
    long len = vec.size();
    for (int i = 0; i < len; ++i) {
        std::cerr << vec(i) << " \n"[i == len - 1];
    }
}
#endif

namespace LPSolver {
    Problem::Problem(size_t n_, size_t m_, const Matrix &A_, const Vector &b_, const Vector &c_)
        : n(n_)
        , m(m_)
        , A(A_)
        , b(b_)
        , c(c_)
        {}

    Position::Position(size_t n_, size_t m_, const Vector &x_, const Vector &y_, const Vector &s_)
        : n(n_)
        , m(m_)
        , x(x_)
        , y(y_)
        , s(s_)
        {}

    double Position::mu() const {
        return x.dot(s) / n;
    }

    double Position::gamma() const {
        return x.cwiseProduct(s).minCoeff() / mu();
    }

    bool Position::isCorrect() const {
        // return std::min(x.minCoeff(), s.minCoeff()) > -1e-6;
        return x.minCoeff() >= 0 && s.minCoeff() > 1e-6;
    }

    Matrix Position::constructInvH() const {
        std::vector<Eigen::Triplet<double>> data;
        for (size_t i = 0; i < n; ++i) {
            data.emplace_back(i, i, x[i] / s[i]);
        }
        Matrix invH(n, n);
        invH.setFromTriplets(data.begin(), data.end());
        return invH;
    }

    Position& Position::operator*=(double val) {
        x *= val;
        y *= val;
        s *= val;
        return *this;
    }

    Position& Position::operator/=(double val) {
        return *this *= (1 / val);
    }

    Position Position::operator/(double val) const {
        return *this * (1 / val);
    }

    Position Position::operator*(double val) const {
        Position copy = *this;
        copy *= val;
        return copy;
    }

    Position& Position::operator+=(const Position &other) {
        x += other.x;
        y += other.y;
        s += other.s;
        return *this;
    }

    Position Position::operator+(const Position &other) const {
        Position copy = *this;
        copy += other;
        return copy;
    }

    Position Position::operator-() const {
        return *this * -1;
    }

    Position& Position::operator-=(const Position &other) {
        x -= other.x;
        y -= other.y;
        s -= other.s;
        return *this;
    }

    Position Position::operator-(const Position &other) const {
        Position copy = *this;
        copy -= other;
        return copy;
    } 
}; // namespace LPSolver
