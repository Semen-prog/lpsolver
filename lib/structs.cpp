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

    Vector Position::get_remaining(const Vector& v) const {
        int remaining = n - index_zero.size() - index_free.size();
        Vector v_remaining(remaining);
        int last = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!index_zero.contains(i) && !index_free.contains(i)) {
                v_remaining(last++) = v(i);
            }
        }
        return v_remaining;
    }

    Vector Position::get_x_remaining() const {
        return get_remaining(x);
    }

    Vector Position::get_s_remaining() const {
        return get_remaining(s);
    }

    int Position::get_n_remaining() const {
        return n - index_zero.size() - index_free.size();
    }

    std::vector<int> Position::get_remaining_indices() const {
        std::vector<int> res;
        for (size_t i = 0; i < n; ++i) {
            if (!index_free.contains(i) && !index_zero.contains(i)) {
                res.emplace_back(i);
            }
        }
        return res;
    }

    std::vector<int> Position::get_free_indices() const {
        std::vector<int> res;
        for (size_t i = 0; i < n; ++i) {
            if (index_free.contains(i)) {
                res.emplace_back(i);
            }
        }
        return res;
    }

    int Position::cnt_free_indices() const {
        return index_free.size();
    }

    double Position::mu() const {
        // return x.dot(s) / n;
        int remaining = n - index_zero.size() - index_free.size();
        Vector x_remaining = get_x_remaining();
        Vector s_remaining = get_s_remaining();
        return x_remaining.dot(s_remaining) / remaining;
    }

    double Position::gamma() const {
        // return x.cwiseProduct(s).minCoeff() / mu();
        return get_x_remaining().cwiseProduct(get_s_remaining()).minCoeff() / mu();
    }

    bool Position::isCorrect() const {
        // return std::min(x.minCoeff(), s.minCoeff()) > -1e-6;
        return x.minCoeff() >= 0 && s.minCoeff() > 1e-6;
    }

    Matrix Position::constructInvH() const {
        Vector x2 = get_x_remaining();
        Vector s2 = get_s_remaining();
        int n2 = get_n_remaining();
        
        std::vector<Eigen::Triplet<double>> data;
        for (int i = 0; i < n2; ++i) {
            data.emplace_back(i, i, x2[i] / s2[i]);
        }
        Matrix invH(n2, n2);
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
