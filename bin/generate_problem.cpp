#include <lpsolver/structs.hpp>
#include <lpsolver/generate.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
    int n, m, random_seed;
    long long max_non_zero;

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " n m max_non_zero random_seed\n";
        return 1;
    }

    n = atoi(argv[1]);
    m = atoi(argv[2]);
    max_non_zero = atoll(argv[3]);
    random_seed = atoi(argv[4]);

    std::pair<LPSolver::Problem, LPSolver::Position> problem = LPSolver::generateProblem(m, n, max_non_zero, random_seed);

    long k = problem.first.A.nonZeros();
    std::cout << n << ' ' << m << ' ' << k << '\n';

    for (int col = 0; col < problem.first.A.outerSize(); ++col) {
        for (int index = problem.first.A.outerIndexPtr()[col]; index < problem.first.A.outerIndexPtr()[col + 1]; ++index) {
            int row = problem.first.A.innerIndexPtr()[index];
            double val = problem.first.A.valuePtr()[index];
            std::cout << row << ' ' << col << ' ' << val << '\n';
        }
    }

    std::cout.precision(20);
    std::cout << std::fixed;

    for (int i = 0; i < m; ++i) {
        std::cout << problem.first.b(i) << " \n"[i == m - 1];
    }

    for (int i = 0; i < n; ++i) {
        std::cout << problem.first.c(i) << " \n"[i == n - 1];
    }

    for (int i = 0; i < n; ++i) {
        std::cout << problem.second.x(i) << " \n"[i == n - 1];
    }

    for (int i = 0; i < m; ++i) {
        std::cout << problem.second.y(i) << " \n"[i == m - 1];
    }

    for (int i = 0; i < n; ++i) {
        std::cout << problem.second.s(i) << " \n"[i == n - 1];
    }

    return 0;
}
