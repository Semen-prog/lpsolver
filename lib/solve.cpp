#include "lpsolver/matrices.hpp"
#include <lpsolver/solver.hpp>

namespace LPSolver {
    void step(const Problem &prob, Position &position, const Delta &delta, double len) {
        position += delta * len;
        Vector nw_s = position.s;
        std::vector<int> zero_indices = position.get_zero_indices();
        Vector add = select_columns(prob.A.transpose(), zero_indices) * position.y;
        for (size_t i = 0; i < zero_indices.size(); ++i) {
            nw_s(zero_indices[i]) = prob.c(zero_indices[i]) - add(i);
        }
        position.s = nw_s;

        double primal = prob.primal_value(position.x);
        double dual = prob.dual_value(position.y);
        if (dual > primal) {
            debug_print("{0}\n", (prob.A.transpose() * position.y + position.s - prob.c).cwiseAbs().maxCoeff());
            debug_print("{0} {1}\n", dual, primal);
            debug_print("{0}\n", (prob.A * position.x - prob.b).cwiseAbs().maxCoeff());
        }
        debug_print("{0} {1}\n", (prob.A.transpose() * position.y + position.s - prob.c).cwiseAbs().maxCoeff(), (prob.A * position.x - prob.b).cwiseAbs().maxCoeff());

        assert((prob.A.transpose() * position.y + position.s - prob.b).cwiseAbs().maxCoeff() < 1e-6);
        assert((prob.A * position.x - prob.b).cwiseAbs().maxCoeff() < 1e-6);
        assert(dual < primal);
    }

    Position solve(const Problem &prob, const Position &init, double eps, double gamma_center, double gamma_predict) {
        int cnt_iter = 0;
        debug_print("Called solve\n");
        Position position = init;
        while (position.mu() > eps) {
            ++cnt_iter;
            while (position.gamma() < gamma_center) {
                Delta delta = centralDirection(prob, position);
                double length = centralLength(position, delta);
                // position += delta * length;
                step(prob, position, delta, length);
                debug_print("center step: mu = {0}, gamma = {1}\n", position.mu(), position.gamma());
            }
            Delta delta = predictDirection(prob, position);
            double length = predictLength(position, delta, gamma_predict);
            // position += delta * length;
            step(prob, position, delta, length);
            debug_print("predict step: mu = {0}, gamma = {1}\n", position.mu(), position.gamma());
            debug_print("INFO 1: {0}\n", position.x.dot(prob.c) - position.y.dot(prob.b));
            debug_print("INFO 2: {0}\n", position.x.cwiseProduct(position.s).cwiseAbs().maxCoeff());
            debug_print("Ax - b max coeff: {0}\n", (prob.A * position.x - prob.b).cwiseAbs().maxCoeff());
        }
        debug_print("iterations count: {0}\n", cnt_iter);
        return position;
    }
} // namespace LPSolver
