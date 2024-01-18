def objectiveG(m):
    obj = 0.0
    M = [A[0][0], 1-A[0][0]]
    N = [B[0][0], 1-B[0][0]]

    for a in range(2):
        for b in range(2):
            obj -= (1+(m-1)*log(1-1/m))*(Z[-1][a][b] - Z[-1][a][b]*M[a]*N[b])
            obj -= 2*log(2)*(1/m*Z[0][a][b] - Z[0][a][b]*M[a]*N[b])
            for i in range(2,m):
                obj -= ((i+1)*log(1+1/i) + (i-1)*log(1-1/i))*(i/m*Z[i-1][a][b] - Z[i-1][a][b]*M[a]*N[b])
    return obj

def objectiveX(X):
    obj = 0.0
    M = [A[0][0], 1-A[0][0]]
    N = [B[0][0], 1-B[0][0]]

    for a in range(2):
        for b in range(2):
            obj -= (X[1]/(X[1]-X[0]))*log(X[1]/X[0])*(X[0]*Z[0][a][b] - Z[0][a][b]*M[a]*N[b])
            obj -= (1+X[-2]*log(X[-2])/(1-X[-2]))*(Z[-1][a][b] - Z[-1][a][b]*M[a]*N[b])
            for i in range(1, len(X)-1):
                obj -= (X[i+1]/(X[i+1]-X[i])*log(X[i+1]/X[i]) - X[i-1]/(X[i]-X[i-1])*log(X[i]/X[i-1]))*(X[i]*Z[i][a][b] - Z[i][a][b]*M[a]*N[b])
    
    return obj

def generateXUniform(m):
    X = [(i+1)/m for i in range(m)]
    return X

def generateXHarmonic(m):
    X = [1/2 - 1/2**(i+2) for i in range(int(m/2)-1)]
    X += [1/2]
    X += [1/2 + 1/2**(i+2) for i in range(int(m/2)-1)][::-1]
    X += [1.0]
    return X

def generateXNM(m):
    X = [1/3**(i+1) for i in range(int((m-1)/2))][::-1]
    X += [0.5]
    X += [1 - 1/3**(i+1) for i in range(int((m-1)/2))] 
    X += [1.0]
    return X

def compute_entropyG():

    ent = 0.0
    sdp.solve()

    ent = -sdp.dual
    
    return (ent-3)/log(2)

def score_constraints(chsh):
    """
    Returns the moment equality constraints for the distribution specified by the
    system sys and the detection efficiency eta. We only look at constraints coming
    from the inputs 0/1. Potential to improve by adding input 2 also?

        sys    --     system parameters
        eta    --     detection efficiency
    """

    constraints = [A[0][0]*B[0][0] + (1-A[0][0])*(1-B[0][0]) + A[0][0]*B[1][0] + (1-A[0][0])*(1-B[1][0]) + A[1][0]*B[0][0] + (1-A[1][0])*(1-B[0][0]) + A[1][0]*(1-B[1][0]) + (1-A[1][0])*B[1][0] - 4*chsh]

    return constraints[:]

def get_subs():
    """
    Returns any substitution rules to use with ncpol2sdpa. E.g. projections and
    commutation relations.
    """
    subs = {}
    # Get Alice and Bob's projective measurement constraints
    subs.update(ncp.projective_measurement_constraints(A,B))

    # Finally we note that Alice and Bob's operators should All commute with Eve's ops
    for a in ncp.flatten([A,B]):
        for z in ncp.flatten(Z):
            subs.update({z*a : a*z})
    
    for z in ncp.flatten(Z):
        subs.update({z*z:z})
        
    return subs

def get_extra_monomials():
    """
    Returns additional monomials to add to sdp relaxation.
    """

    monos = []

    # Add ABZ
    Aflat = ncp.flatten(A)
    Bflat = ncp.flatten(B)
    for a in Aflat:
        for b in Bflat:
            for z in ncp.flatten(Z):
                monos += [a*b*z]
                monos += [a*b]
                monos += [a*z]
                monos += [b*z]

    return monos[:]

import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
import qutip as qtp
from sympy.physics.quantum.dagger import Dagger
import mosek
import chaospy

LEVEL = 1                          # NPA relaxation level
M = 6                              # Number of nodes

A_config = [2,2]
B_config = [2,2]

# Operators in problem
A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
Z = [[ncp.generate_operators('Z'+str(i)+str(a), 2, hermitian=1) for a in range(2)] for i in range(M)]

substitutions = get_subs()             # substitutions used in ncpol2sdpa
moment_ineqs = []                      # moment inequalities
moment_eqs = []                        # moment equalities
op_eqs = []                            # operator equalities
op_ineqs = []                          # operator inequalities
extra_monos = get_extra_monomials()    # extra monomials

ops = ncp.flatten([A,B,Z])        # Base monomials involved in problem
X = generateXNM(M)
print(X)
obj = objectiveX(X)    # Placeholder objective function

chsh_value = 0.75

sdp = ncp.SdpRelaxation(ops, verbose=0, normalized=True, parallel=0)
sdp.get_relaxation(level = LEVEL,
                    equalities = op_eqs[:],
                    inequalities = op_ineqs[:],
                    momentequalities = moment_eqs[:] + score_constraints(chsh_value),
                    momentinequalities = moment_ineqs[:],
                    objective = obj,
                    substitutions = substitutions,
                    extramonomials = extra_monos)

L = []

for c_v in np.linspace(0.75, (sqrt(2)+2)/4, 20):
    sdp.process_constraints(momentequalities=score_constraints(c_v))
    ent = -compute_entropyG()
    #L += [[c_v, ent]]
    print(c_v, ent)

"""
np.savetxt("./data/DI-entropy/CHSH/fr_lin_AB_"+str(M), L)
"""

