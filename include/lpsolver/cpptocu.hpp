#ifndef CPPTOCU_HPP
#define CPPTOCU_HPP

#include <lpsolver/structs.hpp>
#include <vector>

namespace LPSolver {
    enum ALG_TYPE {
        ALG_DEFAULT,
        ALG_1,
        ALG_2
    };
	std::vector<Vector> lu_solve(const Matrix&, const std::vector<Vector>&, ALG_TYPE);
}; // namespace LPSolver;

#endif // CPPTOCU_HPP
