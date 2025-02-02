#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <lpsolver/structs.hpp>

namespace LPSolver {
    
    Delta centralDirection(const Problem &prob, const Position &position);

    double centralLength(const Position &position, const Delta &delta);

    Delta predictDirection(const Problem &prob, const Position &position);

    double predictLength(const Position &position, const Delta &delta, double gamma_predict);

    Position solve(const Problem &prob, const Position &init, double eps, double gamma_center=0.9, double gamma_predict=0.7);

} // namespace LPSolver

#endif // SOLVER_HPP