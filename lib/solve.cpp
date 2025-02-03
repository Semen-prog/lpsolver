#include <lpsolver/solver.hpp>
#include <iostream>

namespace LPSolver {
    Position solve(const Problem &prob, const Position &init, double eps, double gamma_center, double gamma_predict) {
        Position position = init;
        while (position.mu() > eps) {
            while (position.gamma() < gamma_center) {
                Delta delta = centralDirection(prob, position);
                double length = centralLength(position, delta);
                position += delta * length;
#ifdef INFO
                std::cerr << "solve step: mu = " << position.mu() << ", gamma = " << position.gamma() << std::endl;
#endif
            }
            Delta delta = predictDirection(prob, position);
            double length = predictLength(position, delta, gamma_predict);
            position += delta * length;
#ifdef INFO
            std::cerr << "predict step: mu = " << position.mu() << ", gamma = " << position.gamma() << std::endl;
#endif
        }
        return position;
    }
} // namespace LPSolver
