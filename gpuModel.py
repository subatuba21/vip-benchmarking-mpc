import osqp
import numpy as np
import scipy as sp
from scipy import sparse


Q = np.eye(4)
R = np.eye(2) * 0.001

DELTA_T = 0.1
N = 10