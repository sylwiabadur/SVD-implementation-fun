import numpy as np
from scipy.linalg import svd

import matplotlib.pyplot as plt

from random import normalvariate
from PIL import Image

import math
import time

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
        A = AT @ A
        n, m = A.shape
        
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
        print("HEHEH")
        U = np.transpose(Q)
        UT = np.transpose(U)
        # Values @ V = U.T@A => V=inv(Values)@U.T@A
        sigmasInverse = np.linalg.inv(np.diag(sigmas))
        VT = sigmasInverse @ UT @ Aorig
    # elif Norig > Morig: 
    else:
        VT = np.transpose(Q)
        VTT = np.transpose(VT)
        sigmasInverse = np.linalg.inv(np.diag(sigmas))
        # Values @ V = U.T@A => U=A@V@inv(Values)
        U = Aorig @ VTT @ sigmasInverse
    # else:
    #     U = np.transpose(Q)
    #     VT = U
    #     sigmas = np.square(sigmas)
    return U, sigmas, VT


img = Image.open('input/gora.jpeg')
imggray = img.convert('LA')
plt.figure()
plt.imshow(imggray)

imgmat = np.array(list(imggray.getdata(band=0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)
# print(imgmat)
# plt.figure()
# plt.imshow(imgmat, cmap='gray')

# start_time = time.time()
U, sigma, V = np.linalg.svd(imgmat)
# print("--- %s seconds python SVD ---" % (time.time() - start_time))
r = 50

plt.figure()
plt.plot(np.cumsum(sigma)/np.sum(sigma))
plt.title('Cumulative Sum of Sigma Matrix - python SVD')

reconstimg = np.matrix(U[:, :r]) * np.diag(sigma[:r]) * np.matrix(V[:r, :])
plt.figure()
plt.imshow(reconstimg, cmap='gray')

# start_time_mine = time.time()
Umine, sigmaMine, Vmine = svd_simultaneous_power_iteration(imgmat)
# print("--- %s seconds mine SVD---" % (time.time() - start_time_mine))

plt.figure()
plt.plot(np.cumsum(sigmaMine)/np.sum(sigmaMine))
plt.title('Cumulative Sum of Sigma Matrix - python SVD')

reconstimgMine = np.matrix(Umine[:, :r]) * np.diag(sigmaMine[:r]) * np.matrix(Vmine[:r, :])
plt.figure()
plt.imshow(reconstimgMine, cmap='gray')
plt.show()

energy = 0
for i in range(r):
    energy = energy + sigma[i]*sigma[i]
energy = energy / np.sum(np.square(sigma))
print('The first ' + str(r) + ' columns contained ' + str(energy * 100) + '% of the original energy of the image - python SVD')

energyMine = 0
for i in range(r):
    energyMine = energyMine + sigmaMine[i]*sigmaMine[i]
energyMine = energyMine / np.sum(np.square(sigmaMine))
print('The first ' + str(r) + ' columns contained ' + str(energyMine * 100) + '% of the original energy of the image - Mine SVD ')