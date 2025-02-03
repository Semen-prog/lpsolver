#ifndef STRUCTS_HPP
#define STRUCTS_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#ifdef INFO
#include <iostream>
#include <format>
#define debug_print(...) std::print(std::cerr, __VA_ARGS__)
void _printVector(const Eigen::VectorXd&);
#define debug_print_vector(a) debug_print("Vector {0}: ", #a); _printVector(a)
#else
#define debug_print(...)
#define debug_print_vector(a)
#endif

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
