import numpy as np
import qutip as qtp
import chaospy
from matplotlib import pyplot as plt
from math import inf

def rhoSigmaSample(n, epsDich=0.001):
    """
    Returns two density operators rho and sigma sampled at random,
    b such that (1+b)*sigma < b*rho and d = D(rho||sigma)

        n       --  size of the operators
        epsDich --  epsilon used in the dichotomy
    """
    
    sigma = qtp.rand_dm(n)
    rho = qtp.rand_dm(n)

    b = 0.0
    d = qtp.entropy_relative(rho, sigma)

    # Ensures that the density operators sampled have finite relative entropy
    while d == np.inf:  
        sigma = qtp.rand_dm(n)
        rho = qtp.rand_dm(n)
        d = qtp.entropy_relative(rho, sigma)

    # Start of the search for b
    t1 = 1.0
    t2 = 2.0

    # If b is large
    while ((1-t2)*rho + t2*sigma).eigenenergies()[0] >=0:
        t1, t2 = t2, 2*t2

    # If b is small
    while t1 == 1 and ((1-t2)*rho + t2*sigma).eigenenergies()[0] <= 0:
        t2 = (t1+t2)/2

    # Start of the dichotomy
    while t2-t1 > epsDich or t1 == 1.0:
        t = (t1 + t2)/2
        if ((1-t)*rho + t*sigma).eigenenergies()[0] >= 0:
            t1 = t
        else:
            t2 = t
        b = t1-1  #t1-1 is always a lower bound of b since it's a dichotomy
    return rho, sigma, b, d

def quadrature(m):
    """
    Generates nodes and weights of a quadrature rule.

        m   --  order of the quadrature rule (times two for some rules)
    """
    #t, w = chaospy.quadrature.fejer_2(m, (0,1))
    #t, w = chaospy.quadrature.fejer_1(m, (0,1))
    #t, w = chaospy.quadrature.radau(m, chaospy.Uniform(0,1), 1)
    #t, w = chaospy.quadrature.gaussian(m, chaospy.Uniform(0,1))
    #t, w = chaospy.quadrature.legendre_proxy(m, (0,1))
    t, w = chaospy.quadrature.legendre(m, 0, 1)
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
        #The second part of the integral is shifted back to the interval [0;1]
        I += wi*(1/ti * traceMinus(rho - ti*sigma) + 1/(ti+b) * traceMinus((ti/b + 1)*sigma - rho))
    
    return I

N = 1000

L = []

for i in range(1, 20):
    errRel = 0.0
    T, W = quadrature(i)
    for _ in range(N):
        dim = np.random.randint(2,11)
        r, s, B, D1 = rhoSigmaSample(dim)
        D2 = testQuad(B, r, s)
        errRel += abs(D2-D1)/D1*100
    errRel /= N
    print(len(T), errRel)
    L += [[len(T), errRel]]

L = np.array(L)
np.savetxt("./data/quadrature_test/legendre", L)

plt.plot(L[:,0], L[:,1])
plt.savefig("./data/quadrature_test/legendre.png")