#ifndef STRUCTS_HPP
#define STRUCTS_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace LPSolver {
    
    using Matrix = Eigen::SparseMatrix<double>;
    using Vector = Eigen::VectorXd;

    struct Problem {
        size_t n, m;
        Matrix A;
        Vector b, c;

        Problem(size_t, size_t, const Matrix&, const Vector&, const Vector&);
    };

    struct Position {
        size_t n, m;
        Vector x, y, s;

        Position(size_t, size_t, const Vector&, const Vector&, const Vector&);

        double mu() const;

        double gamma() const;

        bool isCorrect() const;

        Matrix constructInvH() const;

        Position& operator*=(double val);

        Position& operator/=(double val);

        Position operator/(double val) const;

        Position operator*(double val) const;

        Position& operator+=(const Position &other);

        Position operator+(const Position &other) const;

        Position operator-() const;

        Position& operator-=(const Position &other);

        Position operator-(const Position &other) const;
    };

    using Delta = Position;

} // namespace LPSover

#endif // STRUCTS_HPP