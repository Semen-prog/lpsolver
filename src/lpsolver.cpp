#include <lpsolver/solver.hpp>
#include <iostream>

int main() {
    std::vector<Eigen::Triplet<double>> data;
    size_t n, m, k; std::cin >> n >> m >> k;
    for (size_t _ = 0; _ < k; ++_) {
        size_t i, j; double val;
        std::cin >> i >> j >> val;
        if (std::abs(val) > 1e-6) data.emplace_back(i, j, val);
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

    /*std::cout << "x = (";
    for (size_t i = 0; i < n; ++i) {
        std::cout << res.x(i);
        if (i < n - 1) std::cout << ", ";
    }
    std::cout << ")\ny = (";
    for (size_t i = 0; i < m; ++i) {
        std::cout << res.y(i);
        if (i < m - 1) std::cout << ", ";
    }
    std::cout << ")\ns = (";
    for (size_t i = 0; i < n; ++i) {
        std::cout << res.s(i);
        if (i < n - 1) std::cout << ", ";
    }*/
    std::cout << ")\n";
}
