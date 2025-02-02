#ifndef SOLVER_HPP
#define SOLVER_HPP

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

    Delta centralDirection(const Problem &prob, const Position &position);

    double centralLength(const Position &position, const Delta &delta);

    Delta predictDirection(const Problem &prob, const Position &position);

    double predictLength(const Position &position, const Delta &delta, double gamma_predict);

    Position solve(const Problem &prob, const Position &init, double eps, double gamma_center=0.9, double gamma_predict=0.7);

} // namespace LPSover

#endif // SOLVER_HPP