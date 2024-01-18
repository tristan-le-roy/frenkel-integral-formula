import numpy as np
import qutip as qtp
import chaospy
from math import pi, sin, cos, log2
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import mosek

def cond_ent(joint, marg):
    """
    Returns H(A|B) = H(AB) - H(B)

    Inputs:
        joint    --     joint distribution on AB
        marg     --     marginal distribution on B
    """

    hab, hb = 0.0, 0.0

    for prob in joint:
        if 0.0 < prob < 1.0:
            hab += -prob*log2(prob)

    for prob in marg:
        if 0.0 < prob < 1.0:
            hb += -prob*log2(prob)

    return hab - hb

def HAgB(sys, eta):
    """
    Computes the error correction term in the key rate for a given system,
    a fixed detection efficiency and noisy preprocessing. Computes the relevant
    components of the distribution and then evaluates the conditional entropy.

        sys    --    parameters of system
        eta    --    detection efficiency
        q      --    bitflip probability
    """

    # Computes H(A|B) required for rate
    [id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
    [theta, a0, a1, b0, b1, b2] = sys[:]
    rho = (cos(theta)*qtp.ket('00') + sin(theta)*qtp.ket('11')).proj()

    # Noiseless measurements
    a00 = 0.5*(id + cos(a0)*sz + sin(a0)*sx)
    b20 = 0.5*(id + cos(b2)*sz + sin(b2)*sx)

    # Alice bins to 0 transforms povm
    A00 = eta * a00 + (1-eta) * id
    A01 = id - A00

    # Bob has inefficient measurement but doesn't bin
    B20 = eta * b20
    B21 = eta * (id - b20)
    B22 = (1-eta) * id

    # joint distribution
    q00 = (rho*qtp.tensor(A00, B20)).tr().real
    q01 = (rho*qtp.tensor(A00, B21)).tr().real
    q02 = (rho*qtp.tensor(A00, B22)).tr().real
    q10 = (rho*qtp.tensor(A01, B20)).tr().real
    q11 = (rho*qtp.tensor(A01, B21)).tr().real
    q12 = (rho*qtp.tensor(A01, B22)).tr().real

    qb0 = (rho*qtp.tensor(id, B20)).tr().real
    qb1 = (rho*qtp.tensor(id, B21)).tr().real
    qb2 = (rho*qtp.tensor(id, B22)).tr().real

    qjoint = [q00,q01,q02,q10,q11,q12]
    qmarg = [qb0,qb1,qb2]

    return cond_ent(qjoint, qmarg)

def objective(ti):
    obj = 0.0
    F = [A[0][0], 1-A[0][0]]
    for a in range(2):
        obj -= ti*Z[a] - Z[a]*F[a]
    return obj

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

def get_subs():
    """
    Returns any substitution rules to use with ncpol2sdpa. E.g. projections and
    commutation relations.
    """
    subs = {}
    # Get Alice and Bob's projective measurement constraints
    subs.update(ncp.projective_measurement_constraints(A,B))

    # Finally we note that Alice and Bob's operators should All commute with Eve's ops
    for a in ncp.flatten([A]):
        for z in Z:
            subs.update({z*a : a*z, Dagger(z)*a : a*Dagger(z)})
    for z in Z:
        subs.update({z*z : z})

    return subs

def get_extra_monomials():
    """
    Returns additional monomials to add to sdp relaxation.
    """

    monos = []

    # Add ABZ
    ZZ = Z + [Dagger(z) for z in Z]
    Aflat = ncp.flatten(A)
    Bflat = ncp.flatten(B)
    for a in Aflat:
        for b in Bflat:
            for z in ZZ:
                monos += [a*b*z]

    # Add monos appearing in objective function
    for z in Z:
        monos += [A[0][0]*Dagger(z)*z]

    return monos[:]

def generate_quadrature(m):
    t, w = chaospy.quadrature.fejner_1(m, (0,1))
    return t[0], w


LEVEL = 2                          # NPA relaxation level
M = 6                              # Number of nodes / 2 in gaussian quadrature
M1 = int(M/4)
M2 = M-M1
T1, W1 = generate_quadrature(M1)      # Nodes, weights of quadrature
T2, W2 = generate_quadrature(M2)
KEEP_M = 0                         # Optimizing mth objective function?
VERBOSE = 0                        # If > 1 then ncpol2sdpa will also be verbose
EPS_M, EPS_A = 1e-4, 1e-4          # Multiplicative/Additive epsilon in iterative optimization

A_config = [2,2]
B_config = [2,2]

A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
Z = ncp.generate_operators('Z', 2, hermitian=1)

substitutions = get_subs()             # substitutions used in ncpol2sdpa
moment_ineqs = []                      # moment inequalities
moment_eqs = []                        # moment equalities
op_eqs = []                            # operator equalities
op_ineqs = []                          # operator inequalities
extra_monos = get_extra_monomials()    # extra monomials

test_sys = [pi/4, 0, pi/2, pi/4, -pi/4, 0]
test_eta = 0.99

ops = ncp.flatten([A,B,Z])        # Base monomials involved in problem
obj = objective(1)    # Placeholder objective function

sdp = ncp.SdpRelaxation(ops, verbose = VERBOSE-1, normalized=True, parallel=0)
sdp.get_relaxation(level = LEVEL,
                    equalities = op_eqs[:],
                    inequalities = op_ineqs[:],
                    momentequalities = moment_eqs[:] + score_constraints(test_sys, test_eta),
                    momentinequalities = moment_ineqs[:],
                    objective = obj,
                    substitutions = substitutions,
                    extramonomials = extra_monos)

for ti in np.linspace(0,10,21).tolist():
    sdp.set_objective(objective(ti))
    sdp.solve()
    print(ti, -sdp.dual)
