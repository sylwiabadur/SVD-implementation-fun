import numpy as np
from scipy.linalg import svd
import math
import csv

#zrodlo http://mlwiki.org/index.php/Power_Iteration

def svd_simultaneous_power_iteration(A, epsilon=0.00001):
    Norig, Morig = A.shape
    # k to mniejszy wymiar
    k = min(Norig,Morig) 

    Aorig = A.copy()
    AT = np.transpose(A)

    if Norig > Morig:
        A = AT @ A
        n, m = A.shape
    elif Norig < Morig:
        A = A @ AT
        n, m = A.shape
    else:
        n,m = Norig, Morig
        
    # nowa macierz o wymiarach n x k i wartosci od 0 do 1
    Q = np.random.rand(n, k) 
    # nowe Q powstaje przez wyznaczenie QR z tej wygenerowanej Q wczesniej
    Q, _ = np.linalg.qr(Q) 
    Qprev = Q

    #blokowa iteracja silowo
    for i in range(100):
        Z = A @ Q
        Q, R = np.linalg.qr(Z)
        err = ((Q - Qprev) ** 2).sum()
        Qprev = Q
        if err < epsilon:
            break     

    # sigma jako pierwiastki kwadratowe z diagonali R macierzy trojkatnej
    R = np.absolute(R)
    sigmas = np.sqrt(np.diag(R)) 

    if Norig < Morig: 
        U = np.transpose(Q)
        UT = np.transpose(U)
        # Values @ V = U.T@A => V=inv(Values)@U.T@A
        sigmasInverse = np.linalg.inv(np.diag(sigmas))
        VT = sigmasInverse @ UT @ Aorig
    elif Norig > Morig: 
        VT = np.transpose(Q)
        VTT = np.transpose(VT)
        sigmasInverse = np.linalg.inv(np.diag(sigmas))
        # Values @ V = U.T@A => U=A@V@inv(Values)
        U = Aorig @ VTT @ sigmasInverse
    else:
        U = np.transpose(Q)
        VT = U
        sigmas = np.square(sigmas)
    return U, sigmas, VT

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
# U, S, V = svd_simultaneous_power_iteration(A)
# print("++++++++++++++++++++")
# print(U)
# print(S)
# print(V)

with open('MAE8-9.csv', 'w', newline='') as file:
    fieldnames = ['iteration', 'MAE']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(20):
        mU = 0
        endMAE = 0
        A = np.random.randint(100, size=(8,9))
        U, S, V = svd_simultaneous_power_iteration(A)
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
        writer.writerow({'iteration': i, 'MAE': endMAE})