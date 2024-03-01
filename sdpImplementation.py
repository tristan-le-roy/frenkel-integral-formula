def objective(t):
    obj = 0.0
    M = [A[0][0], 1-A[0][0]]
    #P = [Z[0][0], 1-Z[0][0]]
    for a in range(len(M)):
        obj += t*Z[a] - Z[a]*M[a]
    return -obj

def objectiveG(m):
    obj = 0.0
    M = [A[0][0], 1-A[0][0]]

    for a in range(2):
        obj -= (1+(m-1)*log(1-1/m))*(Z[0][a] - Z[0][a]*M[a])
        obj -= 2*log(2)*(1/m*Z[1][a] - Z[1][a]*M[a])
        for i in range(2,m):
            obj -= ((i+1)*log(1+1/i) + (i-1)*log(1-1/i))*(i/m*Z[i][a] - Z[i][a]*M[a])
    return obj

"""
def objective1(t):
    obj = 0.0
    M = [A[0][0], 1-A[0][0]]
    for a in range(len(M)):
        obj += Z[a]*M[a] - (t+1)*Z[a]
    return -obj
"""
def compute_entropy():
    ent = 0.0
    
    for i in range(len(W)):
        wi = W[i]
        ti = T[i]
        new_obj = objective(ti)
        sdp.set_objective(new_obj)
        sdp.solve()
        ent += wi/(ti) * -sdp.dual
    """
    for i in range(len(Tp)):
        wi = Wp[i]
        ti = Tp[i]
        new_obj = objective1(ti)
        sdp.set_objective(new_obj)
        sdp.solve()
        print(sdp.dual)
        ent += wi/(ti+1) * -sdp.dual
    """
    return (ent - 1)/log(2)

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

def compute_entropy_linear2(k):
    X = [0]
    X += [0.5 - 0.5**(i+1) for i in range(1,k)]
    X += [0.5]
    X += [0.5 + 0.5**(k-i+1) for i in range(1,k)]
    X += [1]

    new_obj = objective(X[-1])
    sdp.set_objective(new_obj)
    sdp.solve()
    ent = -(1+X[-2]*log(X[-2])/(1-X[-2]))*sdp.dual

    new_obj = objective(X[1])
    sdp.set_objective(new_obj)
    sdp.solve()
    ent -= X[2]*log(X[2]/X[1])/(X[2]-X[1])*sdp.dual

    for i in range(2,2*k):
        new_obj = objective(X[i])
        sdp.set_objective(new_obj)
        sdp.solve()
        ent -= (X[i+1]*log(X[i+1]/X[i])/(X[i+1]-X[i]) - X[i-1]*log(X[i]/X[i-1])/(X[i]-X[i-1]))*sdp.dual
    
    return (ent-1)/log(2)

def print_trace_minus(sdp, sys, eta, m):
    sdp.process_constraints(momentequalities=score_constraints(sys, eta))
    L = [[0.0, 0.0]]
    for i in range(m):
        xi = (i+1)/m
        new_obj = objective(xi)
        sdp.set_objective(new_obj)
        sdp.solve()
        L += [[xi, -sdp.dual/xi]]
    return L

def compute_entropyG():
    ent = 0.0
    sdp.solve()
    
    ent = -sdp.dual
    return (ent-1)/log(2)

def score_constraints(sys, eta=1.0):
    """
    Returns the moment equality constraints for the distribution specified by the
    system sys and the detection efficiency eta. We only look at constraints coming
    from the inputs 0/1. Potential to improve by adding input 2 also?

        sys    --     system parameters
        eta    --     detection efficiency
    """

    # Extract the system
    [id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
    [theta, a0, a1, b0, b1, b2] = sys[:]
    rho = (cos(theta)*qtp.ket('00') + sin(theta)*qtp.ket('11')).proj()

    # Define the first projectors for each of the measurements of Alice and Bob
    a00 = 0.5*(id + cos(a0)*sz + sin(a0)*sx)
    a01 = id - a00
    a10 = 0.5*(id + cos(a1)*sz + sin(a1)*sx)
    a11 = id - a10
    b00 = 0.5*(id + cos(b0)*sz + sin(b0)*sx)
    b01 = id - b00
    b10 = 0.5*(id + cos(b1)*sz + sin(b1)*sx)
    b11 = id - b10

    A_meas = [[a00, a01], [a10, a11]]
    B_meas = [[b00, b01], [b10, b11]]

    constraints = []

    # Add constraints for p(00|xy)
    for x in range(2):
        for y in range(2):
            constraints += [A[x][0]*B[y][0] - (eta**2 * (rho*qtp.tensor(A_meas[x][0], B_meas[y][0])).tr().real + \
                        + eta*(1-eta)*((rho*qtp.tensor(A_meas[x][0], id)).tr().real + (rho*qtp.tensor(id, B_meas[y][0])).tr().real) + \
                        + (1-eta)*(1-eta))]

    # Now add marginal constraints p(0|x) and p(0|y)
    constraints += [A[0][0] - eta * (rho*qtp.tensor(A_meas[0][0], id)).tr().real - (1-eta)]
    constraints += [B[0][0] - eta * (rho*qtp.tensor(id, B_meas[0][0])).tr().real - (1-eta)]
    constraints += [A[1][0] - eta * (rho*qtp.tensor(A_meas[1][0], id)).tr().real - (1-eta)]
    constraints += [B[1][0] - eta * (rho*qtp.tensor(id, B_meas[1][0])).tr().real - (1-eta)]

    return constraints[:]
        

def generate_quadrature(m):
    """
    Generates the Gaussian quadrature nodes t and weights w. Due to the way the
    package works it generates 2*M nodes and weights. Maybe consider finding a
    better package if want to compute for odd values of M.

         m    --    number of nodes in quadrature / 2
    """
    t, w = chaospy.quadrature.fejer_1(m, (0,1))
    #t, w = chaospy.quadrature.gaussian(m, chaospy.Uniform(0, 1))
    #t, w = chaospy.quadrature.radau(m, chaospy.Uniform(0, 1), 1)
    t = t[0]
    return t, w

"""
def generate_quadrature1(m):
    """
"""
    Generates the Gaussian quadrature nodes t and weights w. Due to the way the
    package works it generates 2*M nodes and weights. Maybe consider finding a
    better package if want to compute for odd values of M.

         m    --    number of nodes in quadrature / 2
"""
"""
    t, w = chaospy.quadrature.fejer_1(m, (0,1))
    #t, w = chaospy.quadrature.radau(m, chaospy.Uniform(0, 1), 1)
    t = t[0]
    return t, w
"""

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
        for z in Z:
            subs.update({z*a : a*z})
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
M = 10                              # Number of nodes / 2 in gaussian quadrature
T, W = generate_quadrature(M)      # Nodes, weights of quadrature
#Tp, Wp = generate_quadrature1(M1)

# number of outputs for each inputs of Alice / Bobs devices
# (Dont need to include 3rd input for Bob here as we only constrain the statistics
# for the other inputs).
A_config = [2,2]
B_config = [2,2]
Z_config = [2]

# Operators in problem
A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
Z = ncp.generate_operators('Z', 2, hermitian=1)
#Z = [ncp.generate_operators('Z'+str(i), 2, hermitian=1) for i in range(M)]

substitutions = get_subs()             # substitutions used in ncpol2sdpa
moment_ineqs = []                      # moment inequalities
moment_eqs = []                        # moment equalities
op_eqs = []                            # operator equalities
op_ineqs = []                          # operator inequalities
extra_monos = get_extra_monomials()    # extra monomials

# Defining the test sys
test_sys = [pi/4, 0, pi/2, pi/4, -pi/4, 0]
test_eta = 1.0


ops = ncp.flatten([A,B,Z])        # Base monomials involved in problem
obj = objective(1)    # Placeholder objective function

sdp = ncp.SdpRelaxation(ops, verbose=0, normalized=True, parallel=0)
sdp.get_relaxation(level = LEVEL,
                    equalities = op_eqs[:],
                    inequalities = op_ineqs[:],
                    momentequalities = moment_eqs[:] + score_constraints(test_sys, test_eta),
                    momentinequalities = moment_ineqs[:],
                    objective = obj,
                    substitutions = substitutions,
                    extramonomials = extra_monos)


L = []
for eta in np.linspace(0.8, 1.0, 21)[::-1]:
    sdp.process_constraints(momentequalities=score_constraints(test_sys, eta))
    ent = -compute_entropy()
    L += [[eta, ent]]
    print(eta, ent)

"""
L = []
for i in range(len(ref[:,0])):
    eta = ref[i,0]
    ent_ref = ref[i,1]
    sdp.process_constraints(momentequalities=score_constraints(test_sys, eta))
    ent_fr = -compute_entropy_linear2(M)
    L += [[eta, ent_fr]]
    print(eta, ent_fr, ent_ref)

np.savetxt('./data/DI-entropy/frenkel_linear2_'+str(2*M), L)


for eta in [i/100 for i in range(82,101)]:
    L = print_trace_minus(sdp, test_sys, eta, M)
    np.savetxt('./data/TraceMinus/trmd_'+str(int(100*eta))+'_'+str(M), L)
"""