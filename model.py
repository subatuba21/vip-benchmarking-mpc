from mosek.fusion import *
import numpy as np
import matplotlib.pyplot as plt


N = 10
DELTA_T = 0.1

# goal = [10, 0, 0, 0]
Q = np.eye(4)
R = np.eye(2) * 0.001


def rd(n):
    return round(n, 3)


with Model("double_integrator") as M:
    F = M.parameter([4, 4])
    F_2 = M.parameter([2, 2])

    G = M.parameter([8, 4])
    G_goal = M.parameter(8)

    C = M.parameter([4, 2])
    C_goal = M.parameter(4)

    x_g = M.parameter(4)
    x_start = M.parameter(4)

    x_steps = []
    u_steps = []

    cost = []

    for i in range(0, N):
        x_step = M.variable(4)
        u_step = M.variable(2)

        M.constraint(Expr.sub(Expr.mul(G, x_step), G_goal), Domain.lessThan(0))
        M.constraint(Expr.sub(Expr.mul(C, u_step), C_goal), Domain.lessThan(0))

        x_steps.append(x_step)
        u_steps.append(u_step)

    A = np.array([[1, 0, DELTA_T, 0], [0, 1, 0, DELTA_T], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.array([[0, 0], [0, 0], [DELTA_T, 0], [0, DELTA_T]])

    for i in range(1, N):
        M.constraint(
            Expr.sub(
                x_steps[i],
                Expr.add(Expr.mul(A, x_steps[i - 1]), Expr.mul(B, u_steps[i - 1])),
            ),
            Domain.equalsTo(0),
        )

    M.constraint(Expr.sub(x_steps[0], x_start), Domain.equalsTo(0))

    for i in range(0, N):
        # F = np.linalg.cholesky(Q)
        Fx = Expr.mul(F, Expr.sub(x_g, x_steps[i]))
        r = M.variable()
        M.constraint(Expr.vstack(1, r, Fx), Domain.inRotatedQCone())
        cost.append(r)

        F_u = Expr.mul(F_2, u_steps[i])
        r_u = M.variable()
        M.constraint(Expr.vstack(1, r_u, F_u), Domain.inRotatedQCone())
        cost.append(r_u)

    M.objective("MinimizeDeviationFromGoal", ObjectiveSense.Minimize, Expr.add(cost))

    F.setValue(np.linalg.cholesky(Q))
    F_2.setValue(np.linalg.cholesky(R))

    G.setValue(
        [
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -1],
        ]
    )

    G_goal.setValue(
        [
            30, 30, 30, 30, 5, 5, 5, 5
        ]
    )

    C.setValue([[1, 0], [-1, 0], [0, 1], [0, -1]])
    C_goal.setValue([20, 20, 20, 20])

    x_g.setValue([2.5, 3.5, 0, 0])
    x_start.setValue([2, 3, 2, 3])

    M.solve()

    x_pos = []
    y_pos = []

    labels = []
    for i in range(0, N):
        print("x", i, x_steps[i].level())
        print("u", i, u_steps[i].level())

        x_pos.append(x_steps[i].level()[0])
        y_pos.append(x_steps[i].level()[1])
        labels.append(
            "time: "
            + str(rd(DELTA_T * i))
            + " V_X: "
            + str(rd(x_steps[i].level()[2]))
            + " V_Y: "
            + str(rd(x_steps[i].level()[3]))
        )

    fig, ax = plt.subplots()
    ax.plot(x_pos, y_pos)
    ax.scatter(x_pos, y_pos)

    for i, txt in enumerate(labels):
        ax.text(
            x_pos[i], y_pos[i], txt, ha="right", va="bottom", fontsize=5
        )  # adjust positioning as needed

    plt.show()
