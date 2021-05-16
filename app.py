import tkinter as tk
from tkinter import filedialog, Text, ttk
import os
import numpy as np
import math

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

def stringToNpArray(text):
    rows = text.split(";")
    matrix = []
    for i in rows:
        rowElems = i.split(",")
        nums = []
        for j in rowElems:
            nums.append(float(j))
        matrix.append(nums)
    array = np.array(matrix)
    return array

root = tk.Tk()
root.title("Python SVD implementation")

def calculate():   
    A.delete("1.0","end")
    Amatrix = stringToNpArray(matrix.get())
    print("Your matrix " + matrix.get())
    print(Amatrix)
    A.insert(tk.END, "Your matrix A = \n")
    A.insert(tk.END, np.array2string(Amatrix))

    Umatrix, Sigmamatrix, Vmatrix = svd_simultaneous_power_iteration(Amatrix)
    U.delete("1.0","end")
    U.insert(tk.END, "U = \n")
    U.insert(tk.END, np.array2string(Umatrix))

    Sigma.delete("1.0","end")
    Sigma.insert(tk.END, "Sigma = \n")
    Sigma.insert(tk.END, np.array2string(Sigmamatrix))

    V.delete("1.0","end")
    V.insert(tk.END, "VT = \n")
    V.insert(tk.END, np.array2string(Vmatrix))

matrix = tk.StringVar()  

readText = tk.Button(root, text="Calculate", padx=10, pady=5, fg="white", bg="#263d42", command=calculate)
readText.pack()

lbl = tk.Label(root, text = "Enter the matrix:")
lbl.pack()
matrixEntered = tk.Entry(root, width = 30, textvariable = matrix)
matrixEntered.pack()

A = tk.Text(root, height=8, width=50)
A.pack()
A.insert(tk.END, "Your matrix A = \n")

U = tk.Text(root, height=8, width=50)
U.pack()
U.insert(tk.END, "U = \n")

Sigma = tk.Text(root, height=8, width=50)
Sigma.pack()
Sigma.insert(tk.END, "Sigma = \n")

V = tk.Text(root, height=8, width=50)
V.pack()
V.insert(tk.END, "VT = \n")

root.mainloop()
