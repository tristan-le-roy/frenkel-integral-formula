import numpy as np
import qutip as qtp
import chaospy
from matplotlib import pyplot as plt
from math import inf, floor, ceil, log, log2
import picos

def relative_entropy(r, s):
    res = r.eigenstates()
    ses = s.eigenstates()

    D = 0.0

    for i in range(len(res[0])):
        yi = res[0][i]
        psii = res[1][i]
        for j in range(len(ses[0])):
            xj = ses[0][j]
            phij = ses[1][j]
            D += yi * log(yi/xj) * np.abs((psii.trans().conj()*phij).full()[0][0])**2
    
    return D

def rhoSigmaSample(n):
    """
    Returns two density operators rho and sigma sampled at random,
    b such that (1+b)*sigma < b*rho and d = D(rho||sigma)

        n       --  size of the operators
        epsDich --  epsilon used in the dichotomy
    """
    
    sigma = qtp.rand_dm(n)
    rho = qtp.rand_dm(n)

    d = qtp.entropy_relative(rho, sigma)

    r = qObj2picos(rho)
    s = qObj2picos(sigma)
    FL = picos.Problem()
    t = picos.RealVariable("t", 1)
    FL.add_constraint((1-t)*r + t*s >> 0)
    FL.set_objective("max", t)
    FL.solve()
    lam = FL.value

    return rho, sigma, lam, d

def qObj2picos(r):
    l = r.full()
    n = len(l)
    rho = picos.Constant([[l[i][j] for j in range(n)] for i in range(n)])
    return rho

def traceMinus(rho):
    """
    Computes the sum of the absolute value of the negative eigenvalues of rho
    """

    trM = 0.0
    eigenvalues = rho.eigenenergies()
    for e in eigenvalues:
        if e<0:
            trM -= e
    return trM

def linearApproximation(m, rho, sigma):
    step = 1/(m-1)
    for i in range(1,len(m)-1):
        xi = i*step
        I += ((xi+step)*traceMinus(rho - xi*sigma) - xi*traceMinus(rho - (xi+step)*sigma))*log((xi+step)/xi)
    I += traceMinus(rho-sigma)
    return I


