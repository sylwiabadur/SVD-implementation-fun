# Singular-value decomposition
from numpy import array, matmul, ones, subtract, eye, column_stack, zeros, sort, append, vstack, transpose, split
from scipy.linalg import svd, null_space
from numpy.linalg import matrix_rank, eig
import math


def ownMatrixTranspose(matrix):
    transposed = [list(i) for i in zip(*matrix)]
    return transposed

def calculateR(eigenvalues):
    r = 0
    for i in range(len(eigenvalues)):
        if eigenvalues[i]!=0:
            r+=1
    return r

def createU(squares, A, V, r):
    U = []
    for it in range(r):
        d1 = 1 / squares[it]
        a = A
        b = V[it]
        c = [0]*len(a)

        for i in range(len(a)):
            for j in range(len(a[i])):
                c[i] = c[i] + (a[i][j] * b[j])
        d2 = c
        for x in range(len(d2)):
            d2[x] = d2[x] * d1
        U.append(d2)
    if (r<len(a[0])):
        pozostale = len(a[0]) - r
        U = oblicz_pozostale_u(U,pozostale,r)
    return U

def wektor_razy_liczba(wektor, liczba):
    w = []
    for y in range(len(wektor)):
        w.append(wektor[y] * liczba)
    return w

def oblicz_e(u):
    dl_wektora = 1/len([u])
    e = wektor_razy_liczba(u,dl_wektora)
    return e

def oblicz_pozostale_u(U,pozostale, r):
     Ut = U.transpose()
     U1 = Ut[0]
     e = oblicz_e(U1)
     et = ownMatrixTranspose([e])
     odj = wektor_razy_liczba(U1,(matmul(et,[U1])))
     uj = subtract(e, odj)
     for i in range(1, pozostale+r-1):
        uj = subtract(uj, wektor_razy_liczba(Ut[i], (matmul(Ut[i], et))))
        if i>=r-1:
            Ut.append(uj)
     return Ut

def squaresOfArray(array):
    sqrts = []
    for i in array:
        if i!=0:
            sqrts.append(math.sqrt(abs(i)))
    return sqrts

def makeVArray(egn, ATA, r):
    Varray = []
    for i in range(r):
        v1 = subtract(ATA, egn[i] *eye(r))
        nullspace = null_space(v1)
        Varray.append(nullspace)
    return Varray

#==========================================================

A = array([[2,52,4], [-1, 1,5], [1,2,4]])

#transpozycja macierzy A
At = A.transpose()

#wyznaczenie ATA i AAT
ATA = matmul(At, A) #znana matmultiply to mnozenia macierzy
AAT = matmul(A, At)

#wyznaczenie rzedu dla kazdej z macierzy i wybranie tej z mniejszym

# if (matrix_rank(AAT) > matrix_rank(ATA)):
#     main_matrix = ATA
# else:
#     main_matrix = AAT

# obliczenie wartosci wlasnych
eigevalues, eigenvectors = eig(ATA)
r=len(eigevalues)
print(r)
eigevalues.sort()

print(eigevalues)
print(eigenvectors)

sigmas = squaresOfArray(eigevalues)
sigmas.sort(reverse=True)

print("****")
v = makeVArray(eigevalues, ATA, r)
varr = array(v)

print('Moje U')

# def makeU(sigmas, A, v, r):
#     U = (1/sigmas[0])*matmul(A,v[r-1])
#     for i in range(len(sigmas)-1):
#         element = array((1/sigmas[i+1])*matmul(A,v[r-(i+2)]))
#         print("Element")
#         print(element)
#         eT = transpose(element)
#         print("Element split")
#         heh = split(eT[0],len(element))
#         print(heh)
#         append(U, heh, axis=1)
#     return U

def makeU(sigmas, A, v, r):
    U = []
    for i in range(len(sigmas)):
        element = array((1/sigmas[i])*matmul(A,v[r-(i+1)]))
        U.append(element)
    n = len(U)
    m = len(U[0])
    noweU = []
    for a in range(n):
        newRow = []
        for b in range(m):
            el = U[b][a]
            newRow.append(el)
        noweU.append(newRow)
    return noweU


u1 = (1/sigmas[0])*matmul(A, v[r-1])
print(u1)
u2 = (1/sigmas[1])*matmul(A, v[r-2])
print(u2)
u3 = (1/sigmas[2])*matmul(A, v[r-3])
print(u3)

print("++++++++++++======")
U = makeU(sigmas,A,v,r)
print("MOJE")
print(U)

# # wyznaczenie pierwiastkow kwadratowych z wartosci wlasnych wybranej macierzy wlasciwej
# numrows = len(ATA)
# numcols = len(ATA[0])


# print('Moje sigma \n', sigmas)

# print('Moje V \n',varr)

# # stworzenie macierzy U i uzupelnienie w zaleznosci od rozmiarow macierzy A
# r = calculateR(eigevalues)

# U = createU(pierwiastki, A, v, r)
# U = ownMatrixTranspose(U)

# print("+++++++++++++++++++++++")
# print(U)
# print(E)
# print(v)

print("+++++++++++++++++++=")
# print(A)
# # SVD
Ugot, s, VT = svd(A)
print('SVD U \n', Ugot)
# print('SVD SIGMA: \n ',s)
# print('SVD Vt \n',VT)
