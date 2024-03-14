import numpy as np
from math import log
from scipy import sparse
from scipy.sparse import lil_matrix, csc_matrix
from scipy.optimize import minimize, NonlinearConstraint
from scipy.linalg import logm, sqrtm
from sympy.core.evalf import N
from sympy.physics.quantum import dagger

from ncblockrelaxation_test import *

import sympy
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.dagger import Dagger
from sympy import S
from sympy import sympify

import chaospy
from ncpol2sdpa.nc_utils import flatten

def generateXNM(m):
    X = [1/3**(i+1) for i in range(int((m-1)/2))][::-1]
    X += [0.5]
    X += [1 - 1/3**(i+1) for i in range(int((m-1)/2))] 
    X += [1.0]
    return X

def generate_coeff_integral(X):
    C = [(X[1]/(X[1]-X[0]))*log(X[1]/X[0])]
    for i in range(1, len(X)-1):
        C += [(X[i+1]/(X[i+1]-X[i])*log(X[i+1]/X[i]) - X[i-1]/(X[i]-X[i-1])*log(X[i]/X[i-1]))]
    C += [(1+X[-2]*log(X[-2])/(1-X[-2]))]
    return C

def squashed_entanglement_lb(rho,dim,m=4):

    # Use SDP relaxation to find lower bound on squashed entanglement

    if len(dim) != 2:
        raise Exception("Invalid argument dim: len(dim) != 2")

    dA = dim[0]
    dB = dim[1]

    if dA*dB != len(rho):
        raise Exception("Invalid argument dim: prod(dim) != len(rho)")

    # Compute t_i, w_i
    tvec = generateXNM(m)
    cvec = generate_coeff_integral(tvec)

    print("m = ", m)
    print("  tvec=", tvec)
    print("  cvec=", cvec)

    #Form of the SDP
    Z = [[[Operator('Z'+str(l)+str(i)+str(j)) for j in range(dA)] for i in range(dA)] for l in range(m)]
    ops = flatten(Z)
    
    dagger_substitutions = {Dagger(Z[l][i][j]): Z[l][j][i] for l in range(m) for i in range(dA) for j in range(dA)}
    simplify_func = build_simplify_subs_func([dagger_substitutions])

    def simplify_func_blocks(monb):
        # This function, simplify_func_blocks, can take a pair (b,mon) where
        # mon is a SymPy expression and b is the block
        if isinstance(monb,sympy.Expr):
            return simplify_func(monb)
        elif type(monb) == tuple and len(monb) == 2:
            return (monb[0],simplify_func(monb[1]))
        elif type(monb) == tuple and len(monb) == 3:
            return (monb[0],monb[1],simplify_func(monb[2]))
        else:
            raise Exception("Problem, unknown monb type = ", type(monb), ", monb = ", monb)

    mom_eq = []
    for l in range(m):
        for i in range(dA*dB):
            for j in range(dA*dB):
                for a1 in range(dA):
                    for a2 in range(dA):
                        mom_eq_Z = [((i, j, Z[l][a1][a3]*Z[l][a3][a2]), -1) for a3 in range(dA)]
                        mom_eq_Z += [((i, j, Z[l][a1][a2]), 1)]
                        mom_eq += [mom_eq_Z]
    
    #print("Building custom monomial set")
    monomial_set = []
    # Add monomials of degree 0
    monomial_set += [(b,S.One) for b in range(dA*dB)]
    # Add select monomials of degree 1
    monomial_set += [(b,Z[l][i][x]) for l in range(m) for b in range(dA*dB) for i in range(dA) for x in range(dA)]

    ncbr = NCBlockRelaxationLight(verbose=1)

    ncbr.build_block_moment_matrix(monomial_set, simplify_func=simplify_func_blocks, block=dA*dB, symmetry_action=None)

    moment_subs = {(i,j,S.One): rho[i,j] for i in range(dA*dB) for j in range(dA*dB)}
    print(moment_subs)
    ncbr.do_moment_subs(moment_subs)

    for me in mom_eq:
        ncbr.add_moment_linear_eq(me, 0)
    
    # Set cost function
    obj = np.zeros((dA*dB,dA*dB),dtype=sympy.Expr)

    #print("Forming objective function ...", end="")
    for qq in range(m):
        for i in range(dA):
            for j in range(dA):
                # Tr[rho_{E} Tr_A(Y)]:
                obj -= tvec[qq]*cvec[qq]*Z[qq][i][j]*np.eye(dA*dB)
                for k in range(dB):
                    ik = dB*i+k # in {0,...,dA*dB-1}
                    jk = dB*j+k # in {0,...,dA*dB-1}
                    obj[ik,jk] += cvec[qq]*Z[qq][i][j]
    

    #print("Done.")
    ncbr.create_cost_vector(obj)

    primal, dual, mom_mat, status = ncbr.solve_with_mosek()

    ncbnd = (2*(dA-1) - dual)/log(2)
    print("Lower bound on Squashed entanglement = ", ncbnd)

    return ncbnd


# Werner states
# Returns an array of shape (d,d,d,d)
def werner(p,d):
    W = np.zeros((d,d,d,d))
    I = np.zeros((d,d,d,d))
    F = np.zeros((d,d,d,d)) 
    Psym = np.zeros((d,d,d,d))
    Pasym = np.zeros((d,d,d,d))

    for i in range(d):  
        for j in range(d):
            I[(i,i,j,j)] = 1
            F[(i,j,j,i)] = 1
    Psym = 1/2*(I+F)
    Pasym = 1/2*(I-F)

    W = p*2/(d*(d+1))*Psym + (1-p)*2/(d*(d-1))*Pasym
    return W

# Returns Werner state as a density matrix of size d^2 x d^2
def wernerrho(p,d):
    return werner(p,d).swapaxes(1,2).reshape((d**2,d**2))

if __name__ == "__main__":

    d = 2
    dimA = d
    dimB = d
    p = 0.4
    rho = wernerrho(p, d)
    
    squashed_entanglement_lb(rho,[dimA,dimB],m=2)