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

def DI_sample():
    rhoA = qtp.rand_dm(2)
    rhoE = qtp.rand_dm(2)

    rhoAE = qtp.tensor(rhoA, rhoE)
    IrhoE = qtp.tensor(qtp.qeye(2)/2, rhoE)
    
    d = relative_entropy(rhoAE, IrhoE)
    b = findB(rhoAE, IrhoE)

    return rhoAE, IrhoE, b, d


def qObj2picos(r):
    l = r.full()
    n = len(l)
    rho = picos.Constant([[l[i][j] for j in range(n)] for i in range(n)])
    return rho

def findB(rho, sigma):
    r = qObj2picos(rho)
    s = qObj2picos(sigma)
    FB = picos.Problem()
    t = picos.RealVariable("t", 1)
    FB.add_constraint((1-t)*r + t*s >> 0)
    FB.set_objective("max", t)
    FB.solve()
    return FB.value - 1

def quadrature(m):
    """
    Generates nodes and weights of a quadrature rule.

        m   --  order of the quadrature rule (times two for some rules)
    """
    #t, w = chaospy.quadrature.fejer_2(m, (0,1))
    t, w = chaospy.quadrature.fejer_1(m, (0,1))
    #t, w = chaospy.quadrature.radau(m, chaospy.Uniform(0,1), 1)
    #t, w = chaospy.quadrature.gaussian(m, chaospy.Uniform(0,1))
    #t, w = chaospy.quadrature.legendre_proxy(m, (0,1))
    #t, w = chaospy.quadrature.legendre(m, 0, 1)
    return t[0], w

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

def testQuad(b, rho, sigma):
    """
    Evaluates a quadrature rule of the integral formula of D(rho||sigma)

        b           --  strictly positive real s.t. (1+b)*sigma < b*rho
        rho, sigma  --  density operators
    """

    I = 0.0
    
    for i in range(len(T)):
        wi = W[i]
        ti = T[i]
        I += wi/ti * traceMinus(rho - ti*sigma)
    """
    for i in range(len(Tp)):
        wi   = Wp[i]
        ti = Tp[i]
        I += wi/(ti+b) * traceMinus((1+ti/b)*sigma - rho)
    """
    return I

def linearApproximation(m, rho, sigma):
    step = 1/m
    I = (1 + (m-1)*log(1-1/m))*traceMinus(rho - sigma) + 2*log(2)*traceMinus(rho-sigma/m)
    for i in range(2,m):
        #xi = i*step
        #I += ((xi+step)*traceMinus(rho - xi*sigma) - xi*traceMinus(rho - (xi+step)*sigma))*log((xi+step)/xi)/step
        I += ((i+1)*log(1+1/i) + (i-1)*log(1-1/i))*traceMinus(rho - i/m*sigma)
    #I += traceMinus(rho-sigma)
    return I

r, s, b, D1 = rhoSigmaSample(2)

T,W = quadrature(8)
#Tp,Wp = quadrature(10)

D2 = testQuad(b, r, s)
D3 = linearApproximation(9, r, s)

print(D1, D2, D3)

"""
N = 1000

L = []

for i in range(1, 11):
    errRel = 0.0
    i1 = floor(i/3)
    T, W = quadrature(i-i1)
    Tp, Wp = quadrature(i1+1)
    for _ in range(N):
        #dim = np.random.randint(2,11)
        #r, s, lam, D1 = rhoSigmaSample(dim)
        r, s, b, D1 = DI_sample()
        D2 = testQuad(b, r, s)
        errRel += abs(D2-D1)/D1*100
    errRel /= N
    print(len(T)+len(Tp), errRel)
    L += [[len(T)+len(Tp), errRel]]

L = np.array(L)
np.savetxt("./data/quadrature_test/radau_DI", L)
"""