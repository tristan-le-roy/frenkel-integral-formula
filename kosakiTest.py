import mosek
import qutip as qtp
import numpy as np
import chaospy
from matplotlib import pyplot as plt
import picos

def rhoSigmaSample(n):
    """
    Returns two density operators rho and sigma sampled at random
    and d = D(rho||sigma)

        n       --  size of the operators
        epsDich --  epsilon used in the dichotomy
    """
    
    sigma = qtp.rand_dm(n)
    rho = qtp.rand_dm(n)

    rho.data

    d = qtp.entropy_relative(rho, sigma)

    # Ensures that the density operators sampled have finite relative entropy
    while d == np.inf:  
        sigma = qtp.rand_dm(n)
        rho = qtp.rand_dm(n)
        d = qtp.entropy_relative(rho, sigma)

    return rho, sigma, d

def qObj2picos(r):
    l = r.data.tolil().data
    n = len(l)
    rho = picos.Constant([[l[i][j] for j in range(n)] for i in range(n)])
    return rho

def sdpSolving(r, s, lam, t):
    n = r.dims[0]
    rho = qObj2picos(r)
    sigma = qObj2picos(s)
    sdp = picos.Problem()
    Z = picos.ComplexVariable("Z", rho.shape)
    print(picos.trace(rho))
    sdp.set_objective("min", picos.trace(rho*(Z + Z.Htranspose() + (1-t)*Z.Htranspose()) + sigma*t*Z.Htranspose()).refined)
    sdp.solve()
    print(Z.value)


r, s, d = rhoSigmaSample(3)
sdpSolving(r, s, 0.0, 2.0)




