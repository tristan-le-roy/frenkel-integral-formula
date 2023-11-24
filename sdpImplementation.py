def objective(t):
    obj = 0.0
    M = [A[0][0], 1-A[0][0]]
    #P = [Z[0][0], 1-Z[0][0]]
    for a in range(len(M)):
        obj += t*Z[a] - Z[a]*M[a]
    return -obj

def objectiveG():
    obj = 0.0
    M = [A[0][0], 1-A[0][0]]
    for i in range(len(T)):
        ti = T[i]
        wi = W[i]
        ki = wi/ti
        for a in range(2):
            obj += wi*Z[i][a] - ki * Z[i][a]*M[a]
    return -obj

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
    
    for i in range(len(T)):
        wi = W[i]
        ti = T[i]
        new_obj = objective(ti)
        sdp.set_objective(new_obj)
        sdp.solve()
        ent += wi/ti * -sdp.dual
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
    """
    for t in np.linspace()
    """
    return (ent-1)/log(2)

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
    b20 = 0.5*(id + cos(b2)*sz + sin(b2)*sx)
    b21 = id -b20

    A_meas = [[a00, a01], [a10, a11]]
    B_meas = [[b00, b01], [b10, b11], [b20, b21]]

    constraints = []

    # Add constraints for p(00|xy)
    for x in range(2):
        for y in range(3):
            constraints += [A[x][0]*B[y][0] - (eta**2 * (rho*qtp.tensor(A_meas[x][0], B_meas[y][0])).tr().real + \
                        + eta*(1-eta)*((rho*qtp.tensor(A_meas[x][0], id)).tr().real + (rho*qtp.tensor(id, B_meas[y][0])).tr().real) + \
                        + (1-eta)*(1-eta))]

    # Now add marginal constraints p(0|x) and p(0|y)
    constraints += [A[0][0] - eta * (rho*qtp.tensor(A_meas[0][0], id)).tr().real - (1-eta)]
    constraints += [B[0][0] - eta * (rho*qtp.tensor(id, B_meas[0][0])).tr().real - (1-eta)]
    constraints += [A[1][0] - eta * (rho*qtp.tensor(A_meas[1][0], id)).tr().real - (1-eta)]
    constraints += [B[1][0] - eta * (rho*qtp.tensor(id, B_meas[1][0])).tr().real - (1-eta)]
    constraints += [B[2][0] - eta * (rho*qtp.tensor(id, B_meas[2][0])).tr().real - (1-eta)]

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
        for Zi in Z:
            for z in Zi:
                subs.update({z*a : a*z})
    for Zi in Z:
        for z in ncp.flatten(Zi):
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
            for Zi in Z:
                for z in Zi:
                    monos += [a*b*z]

    return monos[:]

import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
import qutip as qtp
from sympy.physics.quantum.dagger import Dagger
import mosek
import chaospy

LEVEL = 2                          # NPA relaxation level
M = 5                              # Number of nodes / 2 in gaussian quadrature
T, W = generate_quadrature(M)      # Nodes, weights of quadrature
#Tp, Wp = generate_quadrature1(M1)

# number of outputs for each inputs of Alice / Bobs devices
# (Dont need to include 3rd input for Bob here as we only constrain the statistics
# for the other inputs).
A_config = [2,2]
B_config = [2,2,2]
Z_config = [2]

# Operators in problem
A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
Z = [ncp.generate_operators('Z'+str(i), 2, hermitian=1) for i in range(len(T))]

substitutions = get_subs()             # substitutions used in ncpol2sdpa
moment_ineqs = []                      # moment inequalities
moment_eqs = []                        # moment equalities
op_eqs = []                            # operator equalities
op_ineqs = []                          # operator inequalities
extra_monos = get_extra_monomials()    # extra monomials

# Defining the test sys
test_sys = [pi/4, 0, pi/2, pi/4, -pi/4, 0]
test_eta = 1.0


ops = ncp.flatten([A,B,ncp.flatten(Z)])        # Base monomials involved in problem
obj = objectiveG()    # Placeholder objective function

sdp = ncp.SdpRelaxation(ops, verbose=0, normalized=True, parallel=0)
sdp.get_relaxation(level = LEVEL,
                    equalities = op_eqs[:],
                    inequalities = op_ineqs[:],
                    momentequalities = moment_eqs[:] + score_constraints(test_sys, test_eta),
                    momentinequalities = moment_ineqs[:],
                    objective = obj,
                    substitutions = substitutions,
                    extramonomials = extra_monos)


ref = np.loadtxt("./data/reference")

L = []
errRel = 0.0
for i in range(1,len(ref[:,0])):
    eta = ref[i,0]
    ent_ref = ref[i,1]
    sdp.process_constraints(momentequalities=score_constraints(test_sys, eta))
    ent_fr = -compute_entropyG()
    L += [[eta, ent_fr]]
    errRel += abs(ent_fr-ent_ref)/ent_ref*100
    print(eta, ent_fr, ent_ref, abs(ent_fr-ent_ref)/ent_ref*100, abs(ent_fr-ent_ref))

np.savetxt('./data/DI-entropy/frenkel_g_'+str(len(T)), L)
