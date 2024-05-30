import numpy as np
from math import log
from scipy import sparse
from scipy.sparse import lil_matrix, csc_matrix
from scipy.optimize import minimize, NonlinearConstraint
from scipy.linalg import logm, sqrtm
from sympy.core.evalf import N
from sympy.physics.quantum import dagger

from ncblockrelaxation import *

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

def generateXUni(m):
    return [(i+1)/m for i in range(m)]

def generateXpUni(m, lam):
    return [1 + i*(lam-1)/m for i in range(m+1)]

def generate_coeff_integral(X):
    C = [(X[1]/(X[1]-X[0]))*log(X[1]/X[0])]
    for i in range(1, len(X)-1):
        C += [(X[i+1]/(X[i+1]-X[i])*log(X[i+1]/X[i]) - X[i-1]/(X[i]-X[i-1])*log(X[i]/X[i-1]))]
    C += [1 + X[-2]*log(X[-2])/(1-X[-2])]
    return C

def generate_coeff_integral2(X):
    C = [X[1]*log(X[1])/(X[1]-1) - 1]
    for i in range(1,len(X)-1):
        C += [(X[i+1]/(X[i+1]-X[i])*log(X[i+1]/X[i]) - X[i-1]/(X[i]-X[i-1])*log(X[i]/X[i-1]))]
    C += [1 + X[-2]*log(X[-2]/X[-1])/(X[-1]-X[-2])]
    return C

def squashed_entanglement_lb(rho,dim,m=4,solver='mosek'):

    # Use SDP relaxation to find lower bound on squashed entanglement

    if len(dim) != 2:
        raise Exception("Invalid argument dim: len(dim) != 2")

    dA = dim[0]
    dB = dim[1]

    if dA*dB != len(rho):
        raise Exception("Invalid argument dim: prod(dim) != len(rho)")

    # Compute t_i, w_i
    tvec = generateXUni(m)
    tp = generateXpUni(m, dA)
    cvec = generate_coeff_integral(tvec)
    cp = generate_coeff_integral2(tp)
    #tvec[m-1] = 1.0
    #if m == 1:
    #    tvec = np.array([0.5])
    #    wvec = np.array([1.0])
    #elif m == 2:
    #    tvec = np.array([0.25,0.5])
    #    wvec = np.array([1.0,1.0])

    print("m = ", m)
    print("  tvec=", tvec)
    print("  cvec=", cvec)
    print("  tp=", tp)
    print("  cp=", cp)

    # Form the SDP
    Y = [[[Operator('Y'+str(l)+str(i)+str(j)) for j in range(dA)] for i in range(dA)] for l in range(2*m+1)]
    Z = [[[Operator('Z'+str(l)+str(i)+str(j)) for j in range(dA)] for i in range(dA)] for l in range(2*m+1)]
    ops = flatten([Y,Z])
    oporder = {}
    oporder.update({Y[l][i][j]: 1 for i in range(dA) for j in range(dA) for l in range(2*m+1)})
    oporder.update({Z[l][i][j]: 2 for i in range(dA) for j in range(dA) for l in range(2*m+1)})
    
    Dagger_substitutions = {}
    Dagger_substitutions.update({Dagger(Y[l][i][j]): Y[l][j][i] for l in range(2*m+1) for i in range(dA) for j in range(dA)})
    Dagger_substitutions.update({Dagger(Z[l][i][j]): Z[l][j][i] for l in range(2*m+1) for i in range(dA) for j in range(dA)})
    Dagger_substitutions.update({Dagger(Y[l][i][j])*Y[l][x][y]: Y[l][j][i]*Y[l][x][y] for l in range(2*m+1)
                                 for i in range(dA) for j in range(dA) for x in range(dA) for y in range(dA)})
    Dagger_substitutions.update({Dagger(Z[l][i][j])*Z[l][x][y]: Z[l][j][i]*Z[l][x][y] for l in range(2*m+1)
                                 for i in range(dA) for j in range(dA) for x in range(dA) for y in range(dA)})
    
    # Build a function that will simplify any monomial containing Y and Z using commutation relation
    simplify_func = build_simplify_func(ops,oporder,Dagger_substitutions)

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
    for l in range(2*m+1):
        for i in range(dA*dB):
            for j in range(dA*dB):
                for a1 in range(dA):
                    for a2 in range(dA):
                        mom_eq_Y = [((i, j, simplify_func(Y[l][a1][a3]*Y[l][a3][a2])), -1) for a3 in range(dA)]
                        mom_eq_Z = [((i, j, simplify_func(Z[l][a1][a3]*Z[l][a3][a2])), -1) for a3 in range(dA)]
                        mom_eq_Y += [((i, j, simplify_func(Y[l][a1][a2])), 1)]
                        mom_eq_Z += [((i, j, simplify_func(Z[l][a1][a2])), 1)]
                        mom_eq += [mom_eq_Y, mom_eq_Z]

    # Construct symmetry action map that swaps Y and Z
    YZ_replacement_rules = {}
    YZ_replacement_rules.update({Z[l][i][j]: Y[l][i][j] for l in range(2*m+1) for i in range(dA) for j in range(dA)})
    YZ_replacement_rules.update({Y[l][i][j]: Z[l][i][j] for l in range(2*m+1) for i in range(dA) for j in range(dA)})
    def action_YZ_symmetry(monb):
        # Takes a monomial m, and applies replacement Y <-> Z
        if isinstance(monb,sympy.Expr):
            return monb.replace(lambda expr: expr in YZ_replacement_rules,
                                lambda expr: YZ_replacement_rules[expr])
        elif type(monb) == tuple and len(monb) == 2:
            return (monb[0],action_YZ_symmetry(monb[1]))
        elif type(monb) == tuple and len(monb) == 3:
            return (monb[0],monb[1],action_YZ_symmetry(monb[2]))
        else:
            raise Exception("Problem, unknown monb type = ", type(monb), ", monb = ", monb)

    #monomial_set = get_monomials_block(ops, degree=1, simplify_func=simplify_func, block=dA*dB)
    #print("Building custom monomial set")
    monomial_set = []
    # Add monomials of degree 0
    monomial_set += [(b,S.One) for b in range(dA*dB)]
    # Add monomials of degree 1
    for l in range(2*m+1):
        for x in range(dA):
            for i in range(dA):
                for b in range(dA*dB):
                    monomial_set += [(b,Y[l][x][i]), (b,Z[l][x][i])]
    
    
    ncbr = NCBlockRelaxationLight(verbose=1)
    
    ncbr.build_block_moment_matrix(monomial_set, simplify_func=simplify_func_blocks, block=dA*dB, symmetry_action=action_YZ_symmetry)

    moment_subs = {(i,j,S.One): rho[i,j] for i in range(dA*dB) for j in range(dA*dB)}
    ncbr.do_moment_subs(moment_subs)

    for me in mom_eq:
        ncbr.add_moment_linear_eq(me, 0)

    # Set cost function
    obj = np.zeros((dA*dB,dA*dB),dtype=sympy.Expr)
    
    # Tr[rho*Y] = sum_{ijk} Tr[rho_{ik,jk} Y_{ij}]
    #print("Forming objective function ...", end="")
    for qq in range(m):
        for i in range(dA):
            for j in range(dA):
                for k in range(dB):
                    ik = dB*i+k # in {0,...,dA*dB-1}
                    jk = dB*j+k # in {0,...,dA*dB-1}
                    obj[ik,jk] += cvec[qq]*Y[qq][i][j]
                    obj[ik,jk] += cvec[qq]*Z[qq][i][j]
        # Tr[rho_{E} Tr_A(Y)]:
        #   Note that rho_{E} = sum_{i=1}^{dA} sum_{k=1}^{dB} rho_{ik,ik}
        #   and Tr_A(Y) = sum_{l,j=1}^{dA} Y_{lj} Y_{lj}^{\dagger}
        for l in range(dA):
            for j in range(dA):
                obj -= tvec[qq]*cvec[qq]*Y[qq][l][j]*np.eye(dA*dB)
                obj -= tvec[qq]*cvec[qq]*Z[qq][l][j]*np.eye(dA*dB)
    
    
    for qq in range(m,2*m+1):
        for i in range(dA):
            for j in range(dA):
                for k in range(dB):
                    ik = dB*i+k # in {0,...,dA*dB-1}
                    jk = dB*j+k # in {0,...,dA*dB-1}
                    obj[ik,jk] -= cp[qq-m]*Y[qq][i][j]
                    obj[ik,jk] -= cp[qq-m]*Z[qq][i][j]
        # Tr[rho_{E} Tr_A(Y)]:
        #   Note that rho_{E} = sum_{i=1}^{dA} sum_{k=1}^{dB} rho_{ik,ik}
        #   and Tr_A(Y) = sum_{l,j=1}^{dA} Y_{lj} Y_{lj}^{\dagger}
        for l in range(dA):
            for j in range(dA):
                obj += tp[qq-m]*cp[qq-m]*Y[qq][l][j]*np.eye(dA*dB)
                obj += tp[qq-m]*cp[qq-m]*Z[qq][l][j]*np.eye(dA*dB)
    #print("Done.")
    
    ncbr.create_cost_vector(obj)

    # Solve problem
    primal, dual, mom_mat, status = ncbr.solve_with_mosek()

    """
    ncbr.create_cost_vector(obj2)
    primal, dual2, mom_mat, status = ncbr.solve_with_mosek()
    """
    """
    ncbr.create_cost_vector(obj2)
    primal, dual2, mom_mat, status = ncbr.solve_with_mosek()
    """

    """
    # the gamma_{ij} such that [p_{ij} - gamma_{ij} I] is sos
    # these should satisfy np.sum(rho*gamma) == dual
    gamma = x_mat[0][0:(dA*dB),0:(dA*dB)]
    gamma[np.abs(gamma) < 1e-8] = 0.0
    print(gamma)
    print("np.sum(rho*gamma) = ", np.sum(rho*gamma))
    """

    ncbnd = ((dA-1) + (dual)/2)/log(2)
    print("Lower bound on Squashed entanglement = ", ncbnd)

    #for ii in range(len(monomial_set)): print("%s\t%.6e" % (str(monomial_set[ii]),x_mat[ii,ii]))

    #import pdb;
    #pdb.set_trace()

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
    p = 0.0
    rho = wernerrho(p,d)
    
    dimrho = dimA*dimB
    squashed_entanglement_lb(rho,[dimA,dimB],m=6,solver='mosek')
