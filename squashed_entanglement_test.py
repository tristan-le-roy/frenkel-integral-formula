def generate_point_coeff(m):
    #Points used in the integral approximation
    p = [1/3**(i+1) for i in range(int((m-1)/2))][::-1]
    p += [0.5]
    p += [1 - 1/3**(i+1) for i in range(int((m-1)/2))] 
    p += [1.0]

    #Coefficients associated with each points
    c = [(p[1]/(p[1]-p[0]))*log(p[1]/p[0])]
    for i in range(1, m-1):
        c += [(p[i+1]/(p[i+1]-p[i])*log(p[i+1]/p[i]) - p[i-1]/(p[i]-p[i-1])*log(p[i]/p[i-1]))]
    c += [(1+p[-2]*log(p[-2])/(1-p[-2]))]
    return p, c

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

def objective():
    obj = 0.0
    for i in range(M):
        for a1 in range(dA):
            for a2 in range(dA):
                for k in range(dB):
                    a1b = a1*dB + k
                    a2b = a2*dB + k
                    b = k*dB + k 
                    obj += P[i]*Y[a1b][i][a1][a2]*I[a2b]
                    obj += P[i]*Z[a1b][i][a1][a2]*I[a2b]
                    obj -= C[i]*P[i]*Y[b][i][a1][a2]*I[b]
                    obj -= C[i]*P[i]*Z[b][i][a1][a2]*I[b]
                    
    return -obj

def get_substitutions():
    subs = {}
    Yflat = ncp.flatten(Y)
    Zflat = ncp.flatten(Z)
    Iflat = ncp.flatten(I)

    for y in Yflat:
        for z in Zflat:
            for i in Iflat:
                subs.update({z*y: y*z})
                subs.update({i*y: y*i})
                subs.update({i*z: z*i})
    
    for op in Yflat+Zflat+Iflat:
        subs.update({Dagger(op): op})
    
    for i in range(dA*dB):
        for j in range(dA*dB):
            for l in range(M):
                for a1 in range(dA):
                    for a2 in range(dA):
                        subs.update({Y[i][l][a1][a2]*Y[j][l][a1][a2]: Y[i][l][a1][a2]*I[j]})
                        subs.update({Z[i][l][a1][a2]*Z[j][l][a1][a2]: Z[i][l][a1][a2]*I[j]})

    return subs

def score_constraints(rho):
    constraints = []
    for i in range(dA*dB):
        for j in range(dA*dB):
            constraints += [I[i]*I[j] - rho[i,j]]
    return constraints[:]

import numpy as np
from math import log
import mosek
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import Operator

d = 2
dA = d
dB = d
p = 0.0

rho = wernerrho(p, d)

LEVEL = 1
M = 10
P, C = generate_point_coeff(M)

Y = [[[[Operator("(" + str(b)+", Y" + str(i) + str(a1) + str(a2) + ")") for a2 in range(dA)] for a1 in range(dA)] for i in range(M)] for b in range(dA*dB)]
Z = [[[[Operator("(" + str(b)+", Z" + str(i) + str(a1) + str(a2) + ")") for a2 in range(dA)] for a1 in range(dA)] for i in range(M)] for b in range(dA*dB)]
I = [Operator("(" + str(b) + ", 1)") for b in range(dA*dB)]

ops = ncp.flatten([Y,Z,I])

constraints = score_constraints(rho)
subs = get_substitutions()

sdp = ncp.SdpRelaxation(ops, verbose=1, normalized=True)
sdp.get_relaxation(level = LEVEL,
                   objective=objective(),
                   momentequalities=constraints,
                   substitutions=subs)


for p in np.linspace(0.0, 0.6, 25):
    rho = wernerrho(p, d)
    sdp.process_constraints(momentequalities=score_constraints(rho))
    sdp.solve('mosek')

    #print("  d=", d)
    print("  p=", p)
    print("  Lower bound on the squashed entanglement:", sdp.dual)


