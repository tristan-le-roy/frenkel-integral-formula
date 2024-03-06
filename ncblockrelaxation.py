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

def fulltol_vec(F,N):
    assert F.shape[0] == N**2
    diag_lin_ix = [N*i+i for i in range(N)]
    triu_ix = np.triu_indices(N)
    triu_lin_ix = triu_ix[0]*N+triu_ix[1]
    newF = np.sqrt(2)*F.copy()
    newF[diag_lin_ix,:] = newF[diag_lin_ix,:]/np.sqrt(2)
    return newF[triu_lin_ix,:]

def fulltol_vec_blocks(F,block_struct):
    # Input:
    #    an array F of size Nxm where N=N1^2 + ... + Nk^2 where
    #    block_struct = [N1,...,Nk]
    # Output: an array Fvec of size N'xm where
    #        N' = sum_i  Ni*(Ni+1)//2
    # whereby in each matrix we only keep the lower triangular part scaled
    # appropriately by sqrt(2)
    # NOTE: this is the input needed for SCS solver

    row_offsets = [0]
    cumulative_sum = 0
    for block_size in block_struct:
        cumulative_sum += block_size ** 2
        row_offsets.append(cumulative_sum)
    assert F.shape[0] == cumulative_sum
    diag_entries = [row_offsets[k]+i*bs+i for k,bs in enumerate(block_struct)
                                          for i in range(bs) if bs > 1]
    # Lower entries
    lower_entries = [row_offsets[k]+j*bs+i for k,bs in enumerate(block_struct)
                                           for j in range(bs)
                                           for i in range(j+1,bs) if bs > 1]
    # Lower diagonal + diagonal entries
    ld_entries = [row_offsets[k]+j*bs+i for k,bs in enumerate(block_struct)
                                        for j in range(bs)
                                        for i in range(j,bs) if bs > 1]
    
    # Improved version to deal with blocks of size 1
    #Fnew = F.copy()
    #assert isinstance(Fnew,lil_matrix)
    #Fnew[lower_entries,:] = Fnew[lower_entries,:]*np.sqrt(2)


    Fnew = np.sqrt(2)*F.copy()
    #print("Copied F...")
    Fnew[diag_entries,:] = Fnew[diag_entries,:] / np.sqrt(2)
    #print("Scaled diagonal entries, extracting ld entries...")
    return Fnew[ld_entries]


def get_monomials_block(variables, degree, simplify_func=None, block=1):
    """ Generates the list [(i,m)] where i ranges from 0 to block-1, and m
    ranges through all ncmonomials in "variables" of degree at most "degree"
    Note: if block == 1, just returns the list of monomials
    """
    print("Creating list of monomials...")
    ncmonoms = get_monomials(variables, degree)
    if simplify_func is not None:
        # Simplify monomials
        ncmonoms = unique([simplify_func(m) for m in ncmonoms])
    if block == 1:
        return ncmonoms
    else:
        return [(i,m) for m in ncmonoms for i in range(block)]

def build_simplify_comm_func(variables, order):
    """
    A helper function that outputs a function to simplify monomials simply
    based on commutation relations

    INPUTS:
        variables: a list of sympy Operators
        order: a dict assigning a number value to each element of variables
                (raises error if one of the variables has no value)

    OUTPUTS:
        simplify_comm_func: a function that given a monomial in "variables"
            converts it into normal form, by sorting the variables according
            to their order provided by "order"
    """

    # Check first that each variable has an order
    for v in variables:
        try:
            o = order[v]
        except KeyError:
            raise KeyError("Variable ", v, " has no order assigned")

    # Helper function so that getorder(Dagger(v)) and getorder(v**k) returns
    # the same value as order[v]
    def getorder(X):
        #if X in order:
        #    return order[X]
        if isinstance(X,Operator):
            return order[X]
        elif isinstance(X,Dagger) or X.is_Pow:
            return getorder(X.args[0])
        else:
            raise Exception("Unknown type for X= ", X, " (type=", type(X),")")

    def simplify_comm_func(mon):
        if mon == 1:
            return mon
        # Split the monomial into factors, order them, then put them back
        # together
        return sympy.core.mul.Mul(
            *sorted(mon.as_ordered_factors(),key=getorder))

    return simplify_comm_func

def build_simplify_subs_func(subs):
    def simplify_subs_func(mon):
        for sub in subs:
            mon = mon.replace(lambda expr: expr in sub,
                                lambda expr: sub[expr])
        return mon
        
    return simplify_subs_func

def build_simplify_func(variable, order, subs):
    simplify_comm_func = build_simplify_comm_func(variable, order)
    simplify_subs_func = build_simplify_subs_func(subs)
    def simplify_func(mon):
        return simplify_subs_func(simplify_comm_func(mon))
    return simplify_func



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
        
        # Symmetry reduction
        self.Q = None
        self.QQ = None

        # A matrix to reduce the size of the SDP
        self.K = None
        

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
                #monomial_list[k]['loc'].append(N*lv1+lv2)
                #monomial_list[k]['loc'].append(lv1+N*lv2)
        
        # Do the symmetry reduction
        assert len(monomial_list) == n_vars


        # Collect all info about the NCBlockRelaxation:
        self.n_vars = n_vars
        self.monomial_set = monomial_set
        self.monomial_index = monomial_index
        self.monomial_list = monomial_list
        self.simplify_func = simplify_func
        self.block = block
        self.block_struct = [N]

        # SYMMETRY REDUCTION
        if symmetry_action != None:
            print("Before symmetry reduction:")
            print("   n_vars = ", n_vars)
            print("   SDP matrix size = ", N)
            self.symmetry_reduction(symmetry_action)

    def symmetry_reduction(self,action_map):
        """
        Function that performs symmetry reduction on the SDP.
        We're assuming the group acting is Z_2. So it is given by a single function
        called "action_map" that takes a monomial and returns the action of the
        non-identity element on the monomial

        Inputs:
            action_map: function that takes a monomial m and returns transformed
                        monomial. we're assuming that
                            action_map(action_map(m)) == m
                        also it should be compatible with the adjoint transformation, i.e.,
                            action_map(Dagger(m)) == Dagger(action_map(m))
                        
        """

        # new_vars[i] = "old" index of new variable i. len(new_vars) == number of new variables
        new_vars = []
        new_monomial_list = []
        # old_vars_new_ix[ix] = new index of variable with old index ix. len(old_vars_new_ix) == n_vars
        old_vars_new_ix = [-1]*self.n_vars
        transform_map = {}

        # Now, do the symmetry reduction
        # CAN BE VERY SLOW
        # --> Can make it faster by requiring action_map to take a list of monomials instead of a single one...
        # --> the as_ordered_factors() of SymPy is quite fast
        t00 = time.time()
        for ix in range(self.n_vars):
            if ix % 10000 == 0:
                print("Processed %d/%d monomials" % (ix,self.n_vars))
            #mon = sdp_vars[ix]
            mon = self.monomial_list[ix]['mon'][0] # Take one representative
            assert mon != None
            assert action_map(action_map(mon)) == mon, "Action map is not involutive"
            # Apply group action to the monomial
            mon_perm = self.simplify_func(action_map(mon))
            #mon_perm_ix = get_monomial_index(mon_perm,monomial_index,simplify_func=simplify_func,block=block)
            mon_perm_ix = self.monomial_index[mon_perm]
            transform_map[ix] = mon_perm_ix
            if mon_perm_ix >= ix:
                # Add new variable in the symmetry reduced SDP
                new_vars.append(ix)
                # Merge monomial_list[ix] and monomial_list[mon_perm_ix]
                if mon_perm_ix == ix:
                    new_monomial_list.append({'loc': self.monomial_list[ix]['loc'], 'symloc': [],
                                              'mon': self.monomial_list[ix]['mon']})
                else:
                    new_monomial_list.append({'loc': self.monomial_list[ix]['loc'], 'symloc': self.monomial_list[mon_perm_ix]['loc'],
                                              'mon': self.monomial_list[ix]['mon'] + self.monomial_list[mon_perm_ix]['mon']})
                old_vars_new_ix[ix] = len(new_vars)-1
                old_vars_new_ix[mon_perm_ix] = len(new_vars)-1

        # Construct new_monomial_index as the composition of monomial_index with old_vars_new_ix
        new_monomial_index = {mon: old_vars_new_ix[ix] for mon,ix in self.monomial_index.items()}
        new_n_vars = len(new_vars)

        print("DONE time=", time.time() - t00)
        print("Old n_vars = ", self.n_vars)
        print("New n_vars (after symmetry reduction) = ", len(new_vars))

        # Update info
        self.n_vars = new_n_vars
        self.monomial_list = new_monomial_list
        self.monomial_index = new_monomial_index
        
        #print(self.monomial_set)
        
        # BLOCK DIAGONALIZATION -- COMPUTE CHANGE OF BASIS MATRIX
        N = len(self.monomial_set) # size of the moment matrix
        # Q is the orthogonal matrix that will make our SDP matrix block-diagonal
        Q = np.zeros((N,N))
        s0 = []
        s1 = []
        # Construct matrix Q:
        for i,mon in enumerate(self.monomial_set):
            # i = index of monomial, mon = monomial
            if np.abs(Q[i,i]) > 0:
                # this monomial is the permutation of a previously seen monomial
                continue
            # do permutation Y[i][j] <-> Z[i][j]
            mon_perm = self.simplify_func(action_map(mon))
            # find index of new monomial
            j = self.monomial_set.index(mon_perm)
            if i == j:
                # This monomial is fixed by the action
                Q[i,i] = 1
                s0.append(i)
            else:
                # Monomial m is not fixed, it satisfies g.m = m'.
                # Now because our action is Z_2, we know that with
                # v0 = (m+m')/2, and v1 = (m-m')/2 we have:
                #    g.v0 = v0 and g.v1 = -v1
                Q[i,i] = 1/np.sqrt(2)
                Q[i,j] = 1/np.sqrt(2)
                s0.append(i) # irrep 0
                Q[j,j] = 1/np.sqrt(2)
                Q[j,i] = -1/np.sqrt(2)
                s1.append(j) # irrep 1

        # We reorder the rows of Q so that the trivial rep. appears first, and then
        # the sign representation
        N0 = len(s0) # dimension of irrep 0 (trivial representation)
        N1 = len(s1) # dimension of irrep 1 (sign representation)
        assert N0+N1 == N
        # Reorder Q:
        Q = scipy.sparse.csc_matrix(Q[s0+s1,:]) # Is CSC the best option?

        # We will need to compute Q*F_i*Q' for each matrix F_i in our SDP
        # If all the computation was correct, this matrix should be block-diagonal
        # with one block of size N0, and one of size N1
        # For efficiency, note that vec(QXQ') = kron(Q,Q)*vec(X). So we first
        # prepare kron(Q,Q)
        # Note: here X is symmetric, so that it does not matter if vec is row or
        # column-major
        print("Computing kron(Q,Q)...")
        t00 = time.time()
        QQ = scipy.sparse.kron(Q,Q)
        print("Done, time=",time.time() - t00)

        self.Q = Q
        self.QQ = QQ
        self.symmetry_block_struct = [N0,N1]
    


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

        do_block_diag = self.QQ is not None # indicates whether we will do block-diagonalization or not
        num_subs = len(self.moment_subs['vals'])
        num_moment_linear_eqs = len(self.moment_linear_eqs)

        # Compute codimension
        if not do_block_diag:
            m = sum([max(len(e['loc']+e['symloc'])-1,0) for e in self.monomial_list]) + num_subs + num_moment_linear_eqs
            assert m == N*(N+1)//2 - (len(self.monomial_list) - num_subs - num_moment_linear_eqs)
        else:
            N0 = self.symmetry_block_struct[0]
            N1 = self.symmetry_block_struct[1]
            m = sum([max(len(e['loc'])-1,0) for e in self.monomial_list]) + num_subs + num_moment_linear_eqs
            #assert m == N0*(N0+1)//2 + N1*(N1+2)//2 - (len(self.monomial_list) - num_subs)


        #print("N = ", N, ", m = ", m)
        #print("len(monomial_list) - num_subs = ", len(self.monomial_list) - num_subs)

        #import pdb; pdb.set_trace()
        G = lil_matrix((N**2,m),dtype=np.float64)

        # Create F before block diagonalization
        # For each element i in monomial_list, we create precisely len(monomial_list[i]['loc'])-1 matrices F
        # These matrices will enforce that the corresponding entries in Y are equal; each such matrix will have a single 1, and a single -1
        ii = 0
        cons_mons = []
        for e in self.monomial_list:
            el = e['loc']
            if not do_block_diag:
                el = el + e['symloc']
            if len(el) > 1:
                cons_mons.append(e['mon'])
                #print(e)
                loc0 = N*el[0][0] + el[0][1] # sub2ind
                #print("Putting ", len(el)-1, " constraints for monomial ", e['mon'])
                #import pdb
                #pdb.set_trace()
                for j in range(1,len(el)):
                    #print(self.monomial_set[el[j][0]], self.monomial_set[el[j][1]])    
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
            print(G[:,ii])
            ii += 1
        

        print("Finished taking care of moment linear equality constraints")


        # Right-hand side
        c = np.hstack((np.zeros(m-num_subs-num_moment_linear_eqs), self.moment_subs['vals'], moment_lin_eqs_rhs))

        #import pdb
        #pdb.set_trace()

        # Construct F0 --> cost vector
        G0 = lil_matrix((N**2,1))
        for ix in np.flatnonzero(self.cost_vec):
            cval = self.cost_vec[ix]

            # ---- Approach 1 -- not symmetric
            # loc = self.monomial_list[ix]['loc'][0] # Location of monomial in matrix (one of them)
            # # Instead of putting cval in one of the locations only, another option is to put an equal proportion on each loc, i.e.,
            # #    G0[i,j] = cval / (number of locations i,j)
            # # This could be more "robust"?F

            loc = self.monomial_list[ix]['loc'][0]
            loc_ind = N*loc[0] + loc[1] # sub2ind
            G0[loc_ind] = cval
        

        if not do_block_diag:
            # Value of SDP has to be multiplied by -1.0 to get the actual value (because the original problem is a minimization, and here we're maximizing the negative)
            sdpdata = {'F': G, 'F0': G0, 'c': c, 'block_struct': [N], 'postfactor': -1.0, 'cons_mons': cons_mons}

        else:

            # Assuming 2 blocks only for now...
            N0 = self.symmetry_block_struct[0]
            N1 = self.symmetry_block_struct[1]

            # First, ensure that each G_i is symmetric
            triu_ix = np.triu_indices(N,1)
            triu_lin_ix = triu_ix[0]*N+triu_ix[1]
            triu_lin_ix_transpose = triu_ix[1]*N+triu_ix[0]
            assert (G[triu_lin_ix_transpose,:] - G[triu_lin_ix,:]).getnnz() == 0, "Matrices G are not symmetric..."

            # MAIN OPERATION: F_i <- Q*F_i*Q'
            print("Applying Q F_i Q^T for each F_i...")
            t00 = time.time()
            G0_Q = self.QQ @ G0
            G_Q = self.QQ@G # <<-- This line is expensive, takes ~1min
            print("Done, time=",time.time() - t00)

            # Check that the resulting matrices are indeed block-diagonal:
            #offdiag_indices = [N*i+j for j in range(N0,N) for i in range(N0)]
            #assert np.max(np.abs(F0_Q[offdiag_indices,:])) < 1e-6, "Something's wrong: after applying orthogonal transformation matrices are not block diagonal"
            #assert np.max(np.abs(F_Q[offdiag_indices,:])) < 1e-6, "Something's wrong: after applying orthogonal transformation matrices are not block diagonal"

            # Create the block-diagonal program
            block0_indices = [N*i+j for i in range(N0) for j in range(N0)]
            block1_indices = [N*i+j for i in range(N0,N) for j in range(N0,N)]
            G0_block = G0_Q[block0_indices + block1_indices,:]
            G_block = G_Q[block0_indices + block1_indices,:]

            # =======================================
            # TODO: After taking the diagonal blocks of Gs, many of them will be zero! One should remove them:
            # The number of matrices G should drop from m = N*(N+1)//2-n_vars (see above, beginning of function) to N1*(N1+1)//2 + N2*(N2+1)//2 - n_vars !!
            # =======================================

            # Some matrices Gi have 0 blocks, remove them
            # TODO: one should be able to know which ones are actually equal to 0
            G_block_nnz = np.flatnonzero(np.sum(np.abs(G_block),0))
            G_block = G_block[:,G_block_nnz]
            c_new = c[G_block_nnz]
            if np.sum(np.abs(c))-np.sum(np.abs(c_new)) > 1e-4:
                print("Problem, some equations are impossible!")
            else:
                c = c_new


            if self.verbose > 0:
                print("Final number of matrices G_i = ", G_block.shape[1])
                print("Expected number = N0*(N0+1)//2 + N1*(N1+1)//2 - (len(monomial_list)-num_subs) = ", N0*(N0+1)//2 + N1*(N1+2)//2 - (len(self.monomial_list)-num_subs))

            sdpdata = {'F0': G0_block, 'F': G_block, 'c': c, 'block_struct': [N0,N1], 'postfactor': -1.0}
        

        # Identify any trivial rows/columns to remove
        # Check if there are indices i where G0[i,i] = G_k[i,i] = 0 for all k.
        # If this is the case, we can remove row/column from the G's

        oldsdpdata = sdpdata.copy()
        sdpdata = preprocess_sdp_scs(self,sdpdata)

        # row_offsets = [0]
        # cumulative_sum = 0
        # for block_size in sdpdata['block_struct']:
        #     cumulative_sum += block_size ** 2
        #     row_offsets.append(cumulative_sum)
        # assert sdpdata['F'].shape[0] == cumulative_sum
        # diag_entries = [row_offsets[k]+i*bs+i for k,bs in enumerate(sdpdata['block_struct']) for i in range(bs)]
        # diag_mag = (np.abs(sdpdata['F0'][diag_entries]) + np.sum(np.abs(sdpdata['F'][diag_entries,:]),1)).A1 # the result of np.sum on a scipy sparse matrix is of type matrix. Using ".A1" converts it to a flattened array
        # zero_diag = np.argwhere(diag_mag < 1e-7).ravel()
        # pos_diag = np.argwhere(diag_mag > 1e-7).ravel()

        # if self.verbose > 0:
        #     print("Identified ", len(zero_diag), " diagonal elements in F0 and the Fj that are zero and can be removed.")
        #     print(zero_diag)
        #     #print("Not removing them for now...")

        #import pdb;
        #pdb.set_trace()


        return sdpdata
    

    def get_sdp_data(self,form='standard'):
        if form == 'standard':
            # The moment problem is expressed in the standard form, and the SOS problem in inequality (parametrized) form
            sdpdata = self.create_SDP_standard_form()
        else:
            # The other way around
            sdpdata = self.create_SDP_ineq_form()

        # F0 = sdpdata['F0']
        # F = sdpdata['F']
        # c = sdpdata['c']
        # block_struct = sdpdata['block_struct']

        # if len(block_struct) == 1:
        #     # Un-vectorize output for convenience
        #     print("Un-vectorizing SDP data...")
        #     N = block_struct[0]
        #     newdata = {'F0': F0.toarray().flatten().reshape((N,N))}
        #     G = F.toarray()
        #     newdata['F'] = [G[:,i].reshape((N,N)) for i in range(G.shape[1])]
        #     newdata['c'] = sdpdata['c']
        #     return newdata, sdpdata

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
        print(Expr.mul(F_msk,x).toString())

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

        if form == 'standard':
            # X is the moment matrix
            # Dual variable is the SOS Gram matrix
            # The minus sign is because mosek returns a negative definite variable
            moment_matrix = block_diag(*[X[i].level().reshape((block_struct[i],block_struct[i])) for i in range(len(block_struct))])
            print(moment_matrix)
            sos_gram_matrix = -block_diag(*[X[i].dual().reshape((block_struct[i],block_struct[i])) for i in range(len(block_struct))])
        else:
            # X is the sos gram matrix
            # Dual variable is the moment matrix
            moment_matrix = block_diag(*[X[i].dual().reshape((block_struct[i],block_struct[i])) for i in range(len(block_struct))])
            sos_gram_matrix = block_diag(*[X[i].level().reshape((block_struct[i],block_struct[i])) for i in range(len(block_struct))])

        if self.Q is not None:
            # Revert the change of basis matrix
            moment_matrix = self.Q @ moment_matrix @ self.Q.T
            sos_gram_matrix = self.Q @ sos_gram_matrix @ self.Q.T

        assert np.linalg.norm(sos_gram_matrix - sos_gram_matrix.conj().T) < 1e-8, "sos_gram_matrix is not Hermitian"
        assert np.min(np.linalg.eigvalsh(sos_gram_matrix)) > -1e-7, "sos_gram_matrix is not psd"

        assert np.linalg.norm(moment_matrix - moment_matrix.conj().T) < 1e-8, "moment_matrix is not Hermitian"
        assert np.min(np.linalg.eigvalsh(moment_matrix)) > -1e-7, "moment_matrix is not psd"

        #import pdb
        #pdb.set_trace()

        print(moment_matrix)

        return constant_term+postfactor*primal, constant_term+postfactor*dual, moment_matrix, status


    ########################### OLD FUNCTION #################################



def preprocess_sdp_scs(self,sdpdata):

    # Identify diagonal elements that are identically zero, make SDP smaller and set equality constraints on off-diagonal elements
    # TODO... UNFINISHED!!

    print("-------------------")
    print("SDP preprocessing")
    print("-------------------")

    F0 = sdpdata['F0']
    F = sdpdata['F']
    c = sdpdata['c']
    block_struct = sdpdata['block_struct']

    # SDP preprocessing
    row_offsets = [0]
    cumulative_sum = 0
    for block_size in block_struct:
        cumulative_sum += block_size ** 2
        row_offsets.append(cumulative_sum)
    assert F.shape[0] == cumulative_sum
    diag_entries = [row_offsets[k]+i*bs+i for k,bs in enumerate(block_struct) for i in range(bs)]
    diag_mag = (np.abs(F0[diag_entries]) + np.sum(np.abs(F[diag_entries,:]),1)).A1 # the result of np.sum on a scipy sparse matrix is of type matrix. Using ".A1" converts it to a flattened array
    zero_diag = np.argwhere(diag_mag < 1e-7).flatten()
    pos_diag = np.argwhere(diag_mag > 1e-7).flatten()

    if self.verbose > 0:
        print("Identified ", len(zero_diag), " diagonal elements in F0 and the Fj that are zero and can be removed.")

    print("Skipping, not removing any...")
    return sdpdata

    #import pdb
    #pdb.set_trace()

    if len(block_struct) == 1:
        if self.verbose > 0:
            print("Removing them, and adding equality constraints")
        N = block_struct[0]
        indices_to_keep = [N*i+j for i in pos_diag for j in pos_diag]
        newF0_1 = F0[indices_to_keep]
        newF_1 = F[indices_to_keep,:]
        newbs_1 = [len(pos_diag)]

        # Add equality constraints, all off-diagonal entries should be zero
        other_indices = [N*i+j for i in range(N) for j in zero_diag if j > i]
        newF0_2 = F0[other_indices]
        newF_2 = F[other_indices,:]
        newbs_2 = [1 for i in other_indices]

        newF0 = lil_matrix(vstack([newF0_1,newF0_2,-newF0_2]))
        newF = lil_matrix(vstack([newF_1,newF_2,-newF_2]))
        newbs = newbs_1 + newbs_2 + newbs_2

        print("Result of Reduction")
        print("  Changed ", N, "x", N, " SDP to an SDP with a block of size ", len(pos_diag),"x",len(pos_diag)," and ", 2*len(other_indices), " inequality constraints")

        #import pdb
        #pdb.set_trace()

        return {'F0': newF0, 'F': newF, 'c': c, 'block_struct': newbs}

    

        