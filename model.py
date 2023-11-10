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

def x_step_name(i):
    return "x_step_" + str(i)

def u_step_name(i):
    return "u_step_" + str(i)


def CreateModel():
    M = Model("double_integrator")
    F = M.parameter("F", [4, 4])
    F_2 = M.parameter("F_2", [2, 2])

    G = M.parameter("G", [8, 4])
    G_goal = M.parameter("G_goal", 8)

    C = M.parameter("C", [4, 2])
    C_goal = M.parameter("C_goal", 4)

    x_g = M.parameter("x_g", 4)
    x_start = M.parameter("x_start", 4)

    x_steps = []
    u_steps = []

    cost = []

    for i in range(0, N):
        x_step = M.variable(x_step_name(i), 4)
        u_step = M.variable(u_step_name(i), 2)

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
    return M



M = CreateModel()

M.getParameter("F").setValue(np.linalg.cholesky(Q))
M.getParameter("F_2").setValue(np.linalg.cholesky(R))

M.getParameter("G").setValue(
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

M.getParameter("G_goal").setValue(
    [
        30, 30, 30, 30, 5, 5, 5, 5
    ]
)

M.getParameter("C").setValue([[1, 0], [-1, 0], [0, 1], [0, -1]])
M.getParameter("C_goal").setValue([20, 20, 20, 20])

M.getParameter("x_g").setValue([2.5, 3.5, 0, 0])
M.getParameter("x_start").setValue([2, 3, 2, 3])

M.solve()

x_pos = []
y_pos = []

labels = []

for i in range(0, N):
    x_i = M.getVariable(x_step_name(i))
    u_i = M.getVariable(u_step_name(i))
    print("x", i, x_i.level())
    print("u", i, u_i.level())

    x_pos.append(x_i.level()[0])
    y_pos.append(u_i.level()[1])
    labels.append(
        "time: "
        + str(rd(DELTA_T * i))
        + " V_X: "
        + str(rd(x_i.level()[2]))
        + " V_Y: "
        + str(rd(x_i.level()[3]))
    )

fig, ax = plt.subplots()
ax.plot(x_pos, y_pos)
ax.scatter(x_pos, y_pos)

for i, txt in enumerate(labels):
    ax.text(
        x_pos[i], y_pos[i], txt, ha="right", va="bottom", fontsize=5
    )  # adjust positioning as needed

plt.show()
