import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import lil_matrix, csc_matrix, coo_matrix, vstack
from ncpol2sdpa.nc_utils import moment_of_entry, separate_scalar_factor,get_monomials,unique
import sympy
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.dagger import Dagger
from sympy import S
import time
import sys
from scipy.linalg import block_diag

import mosek
from mosek.fusion import *

# ------ SOME HELPER FUNCTIONS ----------

def build_simplify_subs_func(subs):
    def simplify_subs_func(mon):
        for sub in subs:
            mon = mon.replace(lambda expr: expr in sub,
                                lambda expr: sub[expr])
        return mon
    return simplify_subs_func

# ------------------------------------------

class NCBlockRelaxationLight:

    def __init__(self,verbose=True):

        self.moment_subs = None
        self.cost_vec = None

        # List of monomials indexing the moment matrix
        self.monomial_set = None
        # List of monomials arising as u^* v where u,v are in monomial_set
        self.monomial_list = None
        # Dictionary giving the index, in monomial_list, of any monomial of the type u^* v
        self.monomial_index = None

        self.moment_linear_eqs = []
        
        self.verbose = verbose
        

    def build_block_moment_matrix(self, monomial_set, simplify_func=None,
                                symmetry_action=None, block=1):
        """
        Function to build the moment matrix
        INPUTS:
            monomial_sets:
                set of monomials that index the moment matrix
                if block > 1, this should be a list of pairs (i,m) where
                i \in {0,...,block-1}, and m is a monomial
            symmetry_action:
                an action map acting on monomials
            simplify_monomial_func:
                a function which takes as input a monomial, and "simplifies" it
            block:
                block = 1, by default (see monomial_sets)

        OUTPUTS:
            Returns a "NCBlockRelaxation" dict containing the following objects:
                n_vars:
                    number of distinct variables in the moment_matrix
                F:
                    sparse (N^2)x(n_vars-1) lil_matrix, where
                    N = len(monomial_set), and each F[:,k] represents a NxN symmetric
                    matrix. The moment matrix is sum_{k} y_k F[:,k] where y_k is the
                    moment value for the monomial indexed k.
                monomial_index:
                    dict giving the index of each monomial
                    if block == 1, then
                        monomial_index[monomial] = a number in {0,...,n_vars-1}
                    if block > 1, then
                        monomial_index[(i,j,monomial)] = a number in {0,...,n_vars-1}
                block_struct:
                    [N]
        """
        print("Building moment matrix...")
        n_vars = 0
        monomial_index = {}
        monomial_list = []

        N = len(monomial_set)
        print("N (length of monomial_set) = ", N)
        max_n_vars = N*(N+1)//2 # Upper bound on the maximum number of variables

        if simplify_func is None:
            simplify_func = lambda x: x # identity map

        for lv1 in range(N):
            for lv2 in range(lv1+1):
                if block == 1:
                    u, v = monomial_set[lv1], monomial_set[lv2]
                    mom = simplify_func(Dagger(u)*v)
                    momadj = simplify_func(Dagger(v)*u)
                else:
                    i,u = monomial_set[lv1]
                    j,v = monomial_set[lv2]
                    mom = (i,j,simplify_func(Dagger(u)*v)) # Moment <psi_i | u^* v psi_j >
                    momadj = (j,i,simplify_func(Dagger(v)*u))
                    #print("Processing entry (%d,%d), mons ((%d,%s),(%d,%s)): product = %s" % (lv1,lv2,i,u,j,v,mom[2]))
                # Did we already encounter this moment?
                k = -1
                try:
                    k = monomial_index[mom]
                except KeyError:
                    # Maybe we encountered its adjoint
                    try:
                        k = monomial_index[momadj]
                    except KeyError:
                        # Could not find mom or its adjoint, add new variable
                        #print("  Adding new variable")
                        if mom != momadj:
                            monomial_list.append({'loc': [], 'symloc': [], 'mon': [mom,momadj]})
                        else:
                            monomial_list.append({'loc': [], 'symloc': [], 'mon': [mom]})
                        k = n_vars
                        n_vars += 1
                        monomial_index[mom] = k
                        if mom != momadj:
                            # NEW: Add adjoints of monomials also in monomial_index
                            # TODO: Simplify other functions (such as get_monomial_index)
                            monomial_index[momadj] = k
                monomial_list[k]['loc'].append((lv1,lv2))
        
        # Collect all info about the NCBlockRelaxation:
        self.n_vars = n_vars
        self.monomial_set = monomial_set
        self.monomial_index = monomial_index
        self.monomial_list = monomial_list
        self.simplify_func = simplify_func
        self.block = block
        self.block_struct = [N]
    

    def get_monomial_index(self, m):
        """
        Searches for m or its Dagger in monomial_index
        """
        if self.simplify_func is None:
            self.simplify_func = lambda x: x # Set it to be the identity map

        try:
            return self.monomial_index[m]
        except KeyError:
            # Could not find, try its adjoint
            if self.block == 1:
                madj = self.simplify_func(Dagger(m))
            else:
                madj = (m[1],m[0],self.simplify_func(Dagger(m[2])))
            try:
                return self.monomial_index[madj]
            except KeyError:
                return -1


    def do_moment_subs(self, moment_subs):
        """
        Function to substitute some monomials for numerical values

        INPUTS:
            moment_subs:
                a dict of the form {monom: value}
                if block == 1, then monom is a sympy monomial expression
                if block > 1, then monom is a tuple (i,j,m) where i and j are
                integers in {0,...,block-1}, and m is a sympy monomial expression
            ncbr: a "NCBlockRelaxation" dict returned by build_moment_matrix, that
                contains at least the following entries:
                    F:
                        sparse matrix produced by build_moment_matrix
                    monomial_index:
                        monomial index produced by build_moment_matrix
                    block:
                        same as other functions
                    simplify_func

        OUTPUTS:
            None
            It adds two new entries to the ncbr dict, namely:
            F0:
                a sparse matrix of shape (N^2,1), where N = F.shape[0] is the size
                of the moment matrix
            kl:
                a list of monomial indices of the variables that were substituted
                kl satisfies len(kl) <= len(moment_subs)
                (The case len(kl) < len(moment_subs) happens when moment_subs
                contains redundant substitutions)
        """

        # Check first that given substitutions are consistent
        # i.e., no monomial is given twice, and if it is the case, the substitution
        # value is the same
        print("Moment substitutions...")
        kl = []
        klvals = []
        ksubs = {}
        for r,val in moment_subs.items():
            # Find index of monomial
            k = self.get_monomial_index(r)
            if k == -1:
                raise KeyError("Supplied monomial ", r, " could not "
                            "be found in monomial_index")
            if k in kl:
                assert ksubs[k] == val
                #print("Monomial ", r, " appears again with same value, skipping")
                continue
            kl = kl + [k]
            klvals = klvals + [val]
            ksubs.update({k: val})
        
        self.moment_subs = {'indices': kl, 'vals': klvals}


    def add_moment_linear_eq(self, a, val):
        # a is a list of tuple of the form (monomial,coefficient of monomial)
        # val is a scalar
        #print("Adding linear equality in moments")
        kl = []
        for r,coeff in a:
            k = self.get_monomial_index(r)
            if k == -1: raise KeyError("Supplied monomial ", r, " could not be found in monomial_index")
            kl = kl + [(k,coeff)]
        self.moment_linear_eqs += [(kl,val)]


    def create_cost_vector(self, obj):
        """
        Sets the objective function for the relaxation

        INPUTS:
            obj:
                a block x block np.array, where each entry is a sympy expression
            ncbr: A "NCBlockRelaxation" dict that contains at least the following
                entries:
                    n_vars:
                        number of variables in the SDP
                    monomial_index:
                        the monomial_index produced by build_moment_matrix
                    block:
                        same as before

        OUTPUTS:
            None
            Updates the ncbr dict with an entry called 'costvec' containing a numpy
            array of length n_vars
        """
        n_vars = self.n_vars
        monomial_index = self.monomial_index
        block = self.block
        simplify_func = self.simplify_func

        print("Setting objective function")
        if block == 1 and not isinstance(obj,np.array):
            obj = np.array([[obj]])
        assert len(obj.shape) == 2 and obj.shape[0] == block and obj.shape[1] == block

        cost_vec = np.zeros(n_vars)

        for i in range(block):
            assert len(obj[i]) == block
            for j in range(block):
                if obj[i][j] != 0:
                    pij = obj[i][j].expand()
                    # Expand p_{ij} into monomials terms
                    for term in pij.as_ordered_terms():
                        monom,coeff = separate_scalar_factor(term)
                        if np.abs(coeff) < 1e-12:
                            # Treat as zero
                            continue
                        # Find index of (i,j,monom) in monomial_index
                        r = monom if block == 1 else (i,j,monom)
                        k = self.get_monomial_index(r)
                        if k == -1:
                            #import pdb; pdb.set_trace()
                            raise KeyError("Monomial ", r, " in objective function"
                                        " could not be found in monomial_index")
                        # k is index of r
                        cost_vec[k] += coeff

        # Update ncbr
        self.cost_vec = cost_vec
            
    def create_SDP_standard_form(self):

        # FORMULATES PROBLEM IN FORM:
        #
        #      max  -tr(G0*Y) s.t. Y >= 0, tr(Gi Y) = c_i
        #      min   c'x      s.t. G0 + x1 G1 + ... + xn Gn >= 0
        #
        # Returns sdpdata dict containing 'G0','G', 'c', 'block_struct' and 'index_postsub'
        #
        # The moment matrix will be the Y and (not G0+x1 G1 + ... + xn Gn)

        if self.verbose > 0:
            print("Creating problem in standard form")

        # Check first that we have moment_subs and a cost vector
        assert (self.moment_subs is not None or len(self.moment_linear_eqs) > 0) and self.cost_vec is not None

        # Hack so that code doesn't break
        if self.moment_subs == None:
            self.moment_subs = {'indices': [], 'vals': []}

        N = len(self.monomial_set)

        # Count number of matrices Gi we will need
        # NOTE: the number of matrices Gi is the *codimension* of the space of valid moment matrices.
        # The subspace of valid moment matrices has dimension equal to
        #          n_vars = len(monomial_list) - num_subs
        # (See function create_SDP_ineq_form where the Fi are nothing but a basis of this subspace)
        # 
        # Here we are interested in the codimension of this subspace. This codimension depends on the ambient space, i.e.,
        # if we do block-diagonalization or not.
        #
        # Case 1: we do *not* block diagonalization, then for each monomial e that appears in monomial_list we need to impose
        #         len(e['loc'])-1 constraints, to make sure that the value of the monomial is the same across the different e['loc'] locations
        #         Also we need an additional num_subs constraints for the moment substitutions
        #         That's a total of 
        #            sum([max(len(e['loc'])-1,0) for e in self.monomial_list]) + num_subs
        #         As a sanity check, this should be equal to
        #            N*(N+1)//2 - (len(self.monomial_list) - num_subs)
        #
        # Case 2: we *do* block diagonalization: in this case the block diagonalization takes care of the equalities Y_{u,v} = Y_{u',v'} where u' = action(u) and v' = action(v)
        #         so we don't need to add a constraint for these equalities
        #         As a sanity check, the final number of matrices we get should be equal to
        #            N1*(N1+1)//2 + N2*(N2+2)//2 - (len(self.monomial_list) - num_subs)
        #         NOTE: In general it is not equal to this, there is a problem with this computation...

        num_subs = len(self.moment_subs['vals'])
        num_moment_linear_eqs = len(self.moment_linear_eqs)

        # Compute codimension
        m = sum([max(len(e['loc']+e['symloc'])-1,0) for e in self.monomial_list]) + num_subs + num_moment_linear_eqs
        assert m == N*(N+1)//2 - (len(self.monomial_list) - num_subs - num_moment_linear_eqs)


        #print("N = ", N, ", m = ", m)
        #print("len(monomial_list) - num_subs = ", len(self.monomial_list) - num_subs)

        G = lil_matrix((N**2,m),dtype=np.float64)

        # Create F before block diagonalization
        # For each element i in monomial_list, we create precisely len(monomial_list[i]['loc'])-1 matrices F
        # These matrices will enforce that the corresponding entries in Y are equal; each such matrix will have a single 1, and a single -1
        ii = 0
        cons_mons = []
        for e in self.monomial_list:
            el = e['loc']
            el = el + e['symloc']
            if len(el) > 1:
                cons_mons.append(e['mon'])
                loc0 = N*el[0][0] + el[0][1] # sub2ind
                #print("Putting ", len(el)-1, " constraints for monomial ", e['mon'])
                for j in range(1,len(el)):  
                    G[loc0,ii] = 1
                    locj = N*el[j][0] + el[j][1]
                    G[locj,ii] = -1
                    ii += 1


        # Construct G's for the moment substitutions
        for ix,val in zip(self.moment_subs['indices'],self.moment_subs['vals']):
            mloc = self.monomial_list[ix]['loc'][0] # Location of monomial in matrix (one of them)
            mloc_ind = N*mloc[0] + mloc[1] # sub2ind
            G[mloc_ind,ii] = 1*val
            ii += 1

        # Construct G's for moment linear equalities (TODO: merge this with moment substitutions)
        moment_lin_eqs_rhs = []
        for ll in self.moment_linear_eqs:
            av = ll[0] # list of the form [(monomial 1 index,coeff),(monomial 2 index,coeff),...]
            rhs = ll[1]
            #print("Linear equality with av=", av, " , rhs = ", rhs)
            moment_lin_eqs_rhs += [rhs]
            for ix,coeff in av:
                mloc = self.monomial_list[ix]['loc'][0] # Location of monomial in matrix (one of them)
                mloc_ind = mloc[0]*N + mloc[1]
                #print("  ix=", ix, ", coeff=", coeff, " mloc_ind=", mloc_ind, ", mloc_ind_T=", mloc_T_ind)   
                G[mloc_ind,ii] = coeff
            ii += 1
        

        print("Finished taking care of moment linear equality constraints")


        # Right-hand side
        c = np.hstack((np.zeros(m-num_subs-num_moment_linear_eqs), self.moment_subs['vals'], moment_lin_eqs_rhs))

        # Construct F0 --> cost vector
        G0 = lil_matrix((N**2,1))
        for ix in np.flatnonzero(self.cost_vec):
            cval = self.cost_vec[ix]
            loc = self.monomial_list[ix]['loc'][0]
            loc_ind = N*loc[0] + loc[1] # sub2ind
            G0[loc_ind] = cval
        
        # Value of SDP has to be multiplied by -1.0 to get the actual value (because the original problem is a minimization, and here we're maximizing the negative)
        sdpdata = {'F': G, 'F0': G0, 'c': c, 'block_struct': [N], 'postfactor': -1.0, 'cons_mons': cons_mons}

        return sdpdata
    

    def get_sdp_data(self,form='standard'):
        if form == 'standard':
            # The moment problem is expressed in the standard form, and the SOS problem in inequality (parametrized) form
            sdpdata = self.create_SDP_standard_form()
        else:
            # The other way around
            sdpdata = self.create_SDP_ineq_form()

        return sdpdata


    def solve_with_mosek(self,form='standard'):

        if form == 'standard':
            # The moment problem is expressed in the standard form, and the SOS problem in inequality (parametrized) form
            sdpdata = self.create_SDP_standard_form()
        else:
            # The other way around
            sdpdata = self.create_SDP_ineq_form()

        #
        # SDP is max -tr(F0*X) s.t. F*vec(X) == c, X >= 0
        #

        F0 = sdpdata['F0']
        F = coo_matrix(sdpdata['F']) 
        c = sdpdata['c']
        block_struct = sdpdata['block_struct']
        constant_term = 0.0
        postfactor = 1.0
        if sdpdata.get('postfactor') is not None:
            postfactor = sdpdata['postfactor']

        assert F0.shape[0] == F.shape[0] and c.shape[0] == F.shape[1]

        print("Forming MOSEK problem")
        print("  block_struct = ", block_struct)
        print("  # of vars (dual) = ", F.shape[1])

        M = Model("ncprob")

        F = F.transpose()

        F0_msk = F0.toarray().flatten()
        F_msk = Matrix.sparse(F.shape[0],F.shape[1],F.row,F.col,F.data)

        # Create SDP variables
        X = [M.variable(Domain.inPSDCone(bs)) for bs in block_struct]
        # x is a vectorized view of the variable
        x = Expr.vstack([X[i].reshape(block_struct[i]**2) for i in range(len(block_struct))])

        # Linear equality constraint
        M.constraint(Expr.mul(F_msk,x),Domain.equalsTo(list(c)))

        # Objective
        obj = Expr.neg(Expr.dot(F0_msk,x))
        M.objective(ObjectiveSense.Maximize,obj)

        # Solve
        M.setLogHandler(sys.stdout)
        M.solve()
        M.acceptedSolutionStatus(AccSolutionStatus.Anything)

        # Get solution
        status = M.getProblemStatus()
        primal = M.primalObjValue()
        dual = M.dualObjValue()

        # X is the moment matrix
        # Dual variable is the SOS Gram matrix
        # The minus sign is because mosek returns a negative definite variable
        moment_matrix = block_diag(*[X[i].level().reshape((block_struct[i],block_struct[i])) for i in range(len(block_struct))])
        print(moment_matrix)
        sos_gram_matrix = -block_diag(*[X[i].dual().reshape((block_struct[i],block_struct[i])) for i in range(len(block_struct))])

        assert np.linalg.norm(sos_gram_matrix - sos_gram_matrix.conj().T) < 1e-8, "sos_gram_matrix is not Hermitian"
        assert np.min(np.linalg.eigvalsh(sos_gram_matrix)) > -1e-7, "sos_gram_matrix is not psd"

        assert np.linalg.norm(moment_matrix - moment_matrix.conj().T) < 1e-8, "moment_matrix is not Hermitian"
        assert np.min(np.linalg.eigvalsh(moment_matrix)) > -1e-7, "moment_matrix is not psd"

        return constant_term+postfactor*primal, constant_term+postfactor*dual, moment_matrix, status
        