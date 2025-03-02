#ifndef STRUCTS_HPP
#define STRUCTS_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unordered_set>

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
    private:
        Vector get_remaining(const Vector&) const;
    public:
        size_t n, m;
        Vector x, y, s;
        std::unordered_set<int> index_zero;
        std::unordered_set<int> index_free;

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

        Vector get_x_remaining() const;

        Vector get_s_remaining() const;

        std::vector<int> get_remaining_indices() const;

        std::vector<int> get_free_indices() const;

        int get_n_remaining() const;

        int cnt_free_indices() const;

        Position remaining() const;
    };

    using Delta = Position;

} // namespace LPSover

#endif // STRUCTS_HPP
