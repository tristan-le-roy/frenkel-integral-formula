import mosek
import qutip as qtp
import numpy as np
import chaospy
from matplotlib import pyplot as plt
import picos
import cvxpy as cp
from qpsolvers import solve_qp
import ncpol2sdpa as ncp
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.dagger import Dagger
from math import log, e, log2

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
            D += yi * log2(yi/xj) * np.abs((psii.trans().conj() * phij).full()[0][0])**2
    
    return D

def rhoSigmaSample(n):
    """
    Returns two density operators rho and sigma sampled at random
    and d = D(rho||sigma)

        n       --  size of the operators
        epsDich --  epsilon used in the dichotomy
    """
    
    sigma = qtp.rand_dm(n)
    rho = qtp.rand_dm(n)
    d = qtp.entropy_relative(rho, sigma, base=2)
    l = findLam(rho, sigma)

    # Ensures that the density operators sampled have finite relative entropy
    while d == np.inf or d < 0.0001 or l > 2:  
        sigma = qtp.rand_dm(n)
        rho = qtp.rand_dm(n)
        d = qtp.entropy_relative(rho, sigma)
        l = findLam(rho, sigma)

    return rho, sigma, d, l

def DI_sample():

    rhoA = qtp.rand_dm(2)
    rhoE = qtp.rand_dm(2)
    rhoAE = qtp.tensor(rhoA, rhoE)
    IrhoE = qtp.tensor(qtp.qeye(2)/2, rhoE)
    d = relative_entropy(rhoAE, IrhoE)
    l = findLam(rhoAE, IrhoE)

    while d == np.inf or d < 0.0001 or l > 4:
        rhoA = qtp.rand_dm(2)
        rhoE = qtp.rand_dm(2)
        rhoAE = qtp.tensor(rhoA, rhoE)
        IrhoE = qtp.tensor(qtp.qeye(2)/2, rhoE)
        d = relative_entropy(rhoAE, IrhoE)
        l = findLam(rhoAE, IrhoE)

    return rhoAE, IrhoE, d, l

def qObj2picos(r):
    l = r.full()
    n = len(l)
    rho = picos.Constant([[l[i][j] for j in range(n)] for i in range(n)])
    return rho

def qObj2np(r):
    l = r.data.tolil().data
    n = len(l)
    rho = [[l[i][j] for j in range(n)] for i in range(n)]
    return rho


def findLam(r, s):
    rho = qObj2picos(r)
    sigma = qObj2picos(s)
    sdp = picos.Problem()
    t = picos.RealVariable("t", 1)
    sdp.add_constraint((1-t)*rho + t*sigma >> 0)
    sdp.set_objective("max", t)
    sdp.solve()
    return sdp.value

def Ft(x,y,t):
    return y*(x-y)/(t*(x-y)+y)

def DFt(er, es, ti):
    D = 0.0
    for i in range(len(er[0])):
        li = er[0][i]
        Pi = er[1][i]*er[1][i].trans().conj()
        for j in range(len(es[0])):
            mj = es[0][j]
            Qj = es[1][j]*es[1][j].trans().conj()
            D += Ft(mj, li, ti) * (Pi*Qj).tr()
    return D.real

def relativeEntropyKosaki(r, s, l):
    I = 0.0
    eigenr = r.eigenstates()
    eigens = s.eigenstates()
    cm = -(1-l)/(len(T)**2 * log(2))
    for i in range(len(T)-1):
        wi = W[i]
        ti = T[i]
        I -= wi/log(2) * DFt(eigenr, eigens, ti)
    return I+cm
    


def quadrature(m):
    """
    Generates nodes and weights of a quadrature rule.

        m   --  order of the quadrature rule (times two for some rules)
    """
    #t, w = chaospy.quadrature.fejer_2(m, (0,1))
    #t, w = chaospy.quadrature.fejer_1(m, (0,1))
    t, w = chaospy.quadrature.radau(m, chaospy.Uniform(0,1), 1)
    #t, w = chaospy.quadrature.gaussian(m, chaospy.Uniform(0,1))
    #t, w = chaospy.quadrature.legendre_proxy(m, (0,1))
    #t, w = chaospy.quadrature.legendre(m, 0, 1)
    return t[0], w


"""
T, W = quadrature(6)
D1 = 0
D2 = 0

r, s, D1, lam = DI_sample()
D2 = relativeEntropyKosaki(r, s, lam)

print(D1, D2)
"""

N = 2000

L = []

for i in range(1, 11):
    errRel = 0.0
    T, W = quadrature(i)
    for k in range(N):
        #dim = np.random.randint(2,11)
        #r, s, D1, lam = rhoSigmaSample(dim)
        r, s, D1, lam = DI_sample()
        D2 = relativeEntropyKosaki(r, s, lam)
        errRel += abs(D2-D1)/D1*100
        """
        if abs(D2-D1)/D1*100 > 100:
            print(D1, D2, abs(D2-D1)/D1*100, lam)
        """
    errRel /= N
    print(len(T), errRel)
    L += [[len(T), errRel]]


L = np.array(L)
np.savetxt("./data/kosaki/radau", L)
"""

plt.plot(L[:,0], L[:,1])
plt.savefig("./data/kosaki/radau_DI.png")
"""