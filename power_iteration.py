import numpy as np
from scipy.linalg import svd

from random import normalvariate

import math
import time


def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = math.sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]

def svd_dominant_eigen(A, epsilon=1e-10):
    n, m = A.shape
    x = randomUnitVector(min(n,m))
    lastV = None
    currentV = x

    if n > m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / np.linalg.norm(currentV)

        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            print("converged in {} iterations!".format(iterations))
            return currentV


def svd_power_iteration(A, k=None, epsilon=1e-10):
    A = np.array(A, dtype=float)
    n, m = A.shape
        
    svd_so_far = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        matrix_for_1d = A.copy()
        #ubtract previous eigenvector(s) component(s)
        for singular_value, u, v in svd_so_far[:i]:
            matrix_for_1d -= singular_value * np.outer(u, v)
        #here also deal with different dimensions, 
        #otherwise shapes don't match for matrix calculus
        if n > m:
            v = svd_dominant_eigen(matrix_for_1d, epsilon=epsilon)  # next singular vector
            u_unnormalized = A @ v
            sigma = np.linalg.norm(u_unnormalized)  # next singular value
            u = u_unnormalized / sigma
        else:
            u = svd_dominant_eigen(matrix_for_1d, epsilon=epsilon)  # next singular vector
            v_unnormalized = A.T @ u
            sigma = np.linalg.norm(v_unnormalized)  # next singular value
            v = v_unnormalized / sigma

        svd_so_far.append((sigma, u, v))

    singular_values, us, vs = [np.array(x) for x in zip(*svd_so_far)]
    return singular_values, us.T, vs


def MAE(Mcalculated, Msvd):
    MAEsum = 0
    if len(Mcalculated) == len(Msvd):
        for i in range(len(Mcalculated)):
            diff = abs(Mcalculated[i] - Msvd[i])
            if math.isnan(diff):
                MAEsum += Msvd[i]
            else:
                MAEsum += diff
    else:
        return 0
    return MAEsum


# A = np.array([[1,2],[3,4]])
# print("---------------------")
# print(A)
# U,S, V = np.linalg.svd(A)
# print("++++++++++++++++++++")
# print(U)
# print(S)
# print(V)

# print("++++++++++++++++++++")
# Ugot, s, VTrans = svd(A)
# print(Ugot)
# print(s)
# print(VTrans)

for i in range(20):
    mU = 0
    endMAE = 0
    A = np.random.randint(100, size=(3,4))
    U, S, V = np.linalg.svd(A)
    Ugot, s, VTrans = svd(A)
    Ucon = np.concatenate(U)
    Ugotcon = np.concatenate(Ugot)
    mU = MAE(Ucon, Ugotcon)
    mU += MAE(S, s)
    Vcon = np.concatenate(V)
    VTranscon = np.concatenate(VTrans)
    mU += MAE(Vcon, VTranscon)
    endMAE = mU/(len(Ucon)+len(S)+len(Vcon))
    print(endMAE)