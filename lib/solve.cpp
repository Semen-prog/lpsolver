#include <lpsolver/solver.hpp>

namespace LPSolver {
    Position solve(const Problem &prob, const Position &init, double eps, double gamma_center, double gamma_predict) {
        debug_print("Called solve\n");
        Position position = init;
        while (position.mu() > eps) {
            while (position.gamma() < gamma_center) {
                Delta delta = centralDirection(prob, position);
                double length = centralLength(position, delta);
                position += delta * length;
                debug_print("center step: mu = {0}, gamma = {1}\n", position.mu(), position.gamma());
            }
            Delta delta = predictDirection(prob, position);
            double length = predictLength(position, delta, gamma_predict);
            position += delta * length;
            debug_print("predict step: mu = {0}, gamma = {1}\n", position.mu(), position.gamma());
        }
        return position;
    }
} // namespace LPSolver
