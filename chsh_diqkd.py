def objective(t):
    obj = 0.0
    M = [A[0][0], 1-A[0][0]]
    #P = [Z[0][0], 1-Z[0][0]]
    for a in range(len(M)):
        obj += t*Z[a] - Z[a]*M[a]
    return -obj

def compute_entropy_linear(m):

    new_obj = objective(1)
    sdp.set_objective(new_obj)
    sdp.solve()
    ent = -(1+(m-1)*log(1-1/m))*sdp.dual

    new_obj = objective(1/m)
    sdp.set_objective(new_obj)
    sdp.solve()
    ent -= 2*log(2)*sdp.dual

    for i in range(2,m):
        new_obj = objective(i/m)
        sdp.set_objective(new_obj)
        sdp.solve()
        ent -= ((i+1)*log(1+1/i) + (i-1)*log(1-1/i))*sdp.dual
    
    return (ent-1)/log(2)

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
        #for Zi in Z:
        for z in Z:
            subs.update({z*a : a*z})
    #for Zi in Z:
    for z in Z:
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
            #for Zi in Z:
            for z in Z:
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
M = 8                              # Number of nodes / 2 in gaussian quadrature

A_config = [2,2]
B_config = [2,2,2]
Z_config = [2]

# Operators in problem
A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
Z = ncp.generate_operators('Z', 2, hermitian=1)

substitutions = get_subs()             # substitutions used in ncpol2sdpa
moment_ineqs = []                      # moment inequalities
moment_eqs = []                        # moment equalities
op_eqs = []                            # operator equalities
op_ineqs = []                          # operator inequalities
extra_monos = get_extra_monomials()    # extra monomials

ops = ncp.flatten([A,B,Z])        # Base monomials involved in problem
obj = objective(0)    # Placeholder objective function

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

for c_v in np.linspace(0.75, (sqrt(2)+2)/4, 10):
    sdp.process_constraints(momentequalities=score_constraints(c_v))
    ent = -compute_entropy_linear(M)
    L += [c_v, ent]
    print(c_v, ent)

np.savetxt("./data/DI-entropy/CHSH/fr_lin_"+str(M), L)


