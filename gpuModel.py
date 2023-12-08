import osqp
import numpy as np
import scipy as sp
from scipy import sparse


DELTA_T = 0.1
N = 10

umin = np.array([-20, -20])
umax = np.array([20, 20])
xmin = np.array([-30, -30, -5, -5])
xmax = np.array([30, 30, 5, 5])
x0 = np.array([2, 3, 2, 3])

Q = sparse.eye(4)
R = sparse.eye(2) * 0.001

nx, nu = [4, 2]
QN = Q

# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')

# - linear objective
# q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
#                np.zeros(N*nu)])

# No linear objective
q = np.zeros((N + 1) * nx + N * nu)

Ad = sparse.csc_matrix([[1, 0, DELTA_T, 0], [0, 1, 0, DELTA_T], [0, 0, 1, 0], [0, 0, 0, 1]])
Bd = sparse.csc_matrix([[0, 0], [0, 0], [DELTA_T, 0], [0, DELTA_T]])

Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N*nx)])
ueq = leq
# - input and state constraints
Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])


prob = osqp.OSQP()
prob.setup(P, q, A, l, u, warm_start=True)

res = prob.solve()
print(res)
