#include <lpsolver/cpptocu.hpp>

namespace LPSolver {
    Delta predictDirection(const Problem &prob, const Position &position) {
        Matrix invH = position.constructInvH();
        Matrix AT = prob.A.transpose();

        Vector dy = lu_solve(prob.A * invH * AT, prob.b);
        Vector ds = -AT * dy;
        Vector dx = -invH * ds - position.x;

        return Delta(position.n, position.m, dx, dy, ds);
    }

    double predictLength(const Position &position, const Delta &delta, double gamma_predict) {
        double step = 1e-3;

        auto ok = [&](double x) {
            return (position + delta * x).isCorrect() && (position + delta * x).gamma() >= gamma_predict;
        };

        while (!ok(step)) {
            step /= 2;
        }

        while (ok(step * 2)) {
            step *= 2;
        }

        double left = step, right = step * 2;
        while (right - left > 1e-2) {
            double mid = (left + right) / 2;
            if (ok(mid)) left = mid;
            else right = mid;
        }
        return left;
    }
};
