from mosek.fusion import *
import numpy as np


N = 10

# goal = [10, 0, 0, 0]
Q = np.eye(4)
R = np.eye(2)

with Model("double_integrator") as M:
    # x = M.variable("x", t) # [Vx, Vy, X, Y]
    # y = M.variable("y", t) # [Vx, Vy, X, Y]
    # vx = M.variable("vx", t)
    # vy = M.variable("vy", t)
    # ux = M.variable("ux", t - 1)
    # uy = M.variable("uy", t - 1)

    # for i in range(t-1):
    #     M.constraint(Expr.sub(x.index(i+1), x.index(i)), Domain.equalsTo(vx.index(i)))
    #     M.constraint(Expr.sub(y.index(i+1), y.index(i)), Domain.equalsTo(vy.index(i)))
    #     M.constraint(Expr.sub(vx.index(i+1), vx.index(i)), Domain.equalsTo(ux.index(i)))
    #     M.constraint(Expr.sub(vy.index(i+1), vy.index(i)), Domain.equalsTo(uy.index(i)))

    # diff_vx = [Expr.sub(vx.index(i), goal[0]) for i in range(t)]
    # diff_vy = [Expr.sub(vy.index(i), goal[1]) for i in range(t)]
    # diff_x = [Expr.sub(x.index(i), goal[2]) for i in range(t)]
    # diff_y = [Expr.sub(y.index(i), goal[3]) for i in range(t)]


    # sum_vx = Expr.add(diff_vx)
    # sum_vy = Expr.add(diff_vy)
    # sum_y = Expr.add(diff_y)
    # sum_x = Expr.add(diff_x)

    # M.objective('MinimizeDeviationFromGoal', ObjectiveSense.Minimize, sum_vx + sum_vy + sum_x + sum_y)

    F = M.parameter([4, 4])
    F_2 = M.parameter([2, 2])
    x_g = M.parameter(4)

    x_steps = []
    u_steps = []

    cost = []

    for i in range(0, N):
        x_steps.append(M.variable(4))
        u_steps.append(M.variable(2))

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


    M.objective('MinimizeDeviationFromGoal', ObjectiveSense.Minimize, Expr.add(cost))

    F.setValue(np.linalg.cholesky(Q))
    F_2.setValue(np.linalg.cholesky(R))

    M.solve()

    for i in range(0, N):
        print("x", i, x_steps[i].level())
        print("u", i, u_steps[i].level())


    








    
