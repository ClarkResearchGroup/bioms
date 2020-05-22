import sys
import copy

import numpy as np
import numpy.linalg as nla
import scipy.sparse as ss
import scipy.sparse.linalg as ssla

from qosy.operatorstring import opstring
from qosy.basis          import Basis, Operator
from qosy.algebra        import structure_constants, _operation_opstring

# TODO: document
def arg(args, arg_name, default_value):
    """Return a parameter from the dictionary
       with the given name if it is exists. Otherwise,
       return the default parameter value.
    """
    
    if arg_name not in args or args[arg_name] is None:
        return default_value
    else:
        return args[arg_name]

# TODO: document
def compute_overlap_inds(vec1, vec1_basis_inds, vec2, vec2_basis_inds):
    """Function to compute the overlap between two Operators,
    whose information is provided as vectors of coefficients
    and the indices of the operator strings in the explored basis
    corresponding to those coefficients.
    """
    
    vec2_dict = dict()
    for ind_v2 in range(len(vec2_basis_inds)):
        vec2_dict[vec2_basis_inds[ind_v2]] = ind_v2

    inds_shared_v1 = []
    inds_shared_v2 = []
    for ind_v1 in range(len(vec1_basis_inds)):
        ind_basis = vec1_basis_inds[ind_v1]
        if ind_basis in vec2_dict:
            inds_shared_v1.append(ind_v1)
            inds_shared_v2.append(vec2_dict[ind_basis])
    inds_shared_v1 = np.array(inds_shared_v1, dtype=int)
    inds_shared_v2 = np.array(inds_shared_v2, dtype=int)

    if len(inds_shared_v1) == 0 or len(inds_shared_v2) == 0:
        return 0.0
    else:
        # Both vectors should be real.
        return np.dot(np.conj(vec1[inds_shared_v1]), vec2[inds_shared_v2])

def _basis_index_convertors(basis, explored_basis):
    # Helper function for building a list and dictionary
    # for converting between the index of an OperatorString
    # in basis and the index of the OperatorString in explored_basis.

    # A list of indices in the explored basis: ind_exp = inds_basis_to_exp[ind]
    inds_basis_to_exp = [explored_basis.index(os) for os in basis]

    # A dictionary of indices mapping the explored basis to basis: ind = inds_exp_to_basis[ind_exp]
    inds_exp_to_basis = dict()
    for ind_b in range(len(basis)):
        inds_exp_to_basis[inds_basis_to_exp[ind_b]] = ind_b

    return [inds_basis_to_exp, inds_exp_to_basis]

def _update_explored_s_constants(inds_exp_basisA, inds_exp_basisB, explored_basis, explored_extended_basis, explored_s_constants, operation_mode):
    # Helper function for updating the data saved in explored_s_constants.
    
    # 1. Find keys that are not in explored_s_constants.
    unexplored_keys = [(ind_exp_B, ind_exp_A)
                       for ind_exp_B in inds_exp_basisB
                       for ind_exp_A in inds_exp_basisA
                       if (ind_exp_B, ind_exp_A) not in explored_s_constants]
    
    # 2. Update explored_s_constants with the new data.
    # (slow: this is the bottleneck of build_s_constants(), computing
    # and saving all of the structure constants. But after these are
    # stored, future runs will save time.)
    for (ind_exp_B, ind_exp_A) in unexplored_keys:
        keyBA           = (ind_exp_B, ind_exp_A)
        (coeff_C, os_C) = _operation_opstring(explored_basis[ind_exp_B], explored_basis[ind_exp_A], operation_mode=operation_mode)

        if os_C is not None:
            explored_extended_basis += os_C
            ind_exp_C                = explored_extended_basis.index(os_C)
            result_C                 = (coeff_C, ind_exp_C)
        else:
            result_C = None
            
        explored_s_constants[keyBA] = result_C

def _build_s_constants_data(basisA, basisB, inds_basisA_to_exp, inds_basisB_to_exp, explored_extended_basis, explored_s_constants):
    # Builds the structure constants given fully updated
    # explored bases and explored_s_constants data.
    # Stores the results as lists of row_inds, col_inds,
    # and data for later use in creating a scipy.csr_matrix l_matrix.

    # The identity OperatorString (needed to identify its index in the extended basis).
    if len(basisA) > 0:
        identity = opstring('I', basisA[0].op_type)
    else:
        identity = opstring('I', 'Pauli')

    # The list of ints mappings indices in basisC (extended_basis) to explored_extended_basis
    # so that inds_basisC_to_exp[ind_C] = ind_exp_C.
    inds_basisC_to_exp = []
    
    # The dictionary mapping indices in explored_extended_basis to basisC (extended_basis)
    # so that inds_exp_to_basisC[ind_exp_C] = ind_C
    # (This is just used in this function to avoid having to call extended_basis.index(os_C)
    # many times for the same OperatorString os_C. This is good to realize: many of the os_C
    # are actually the same.)
    inds_exp_to_basisC = dict()

    # A running index used to keep track of the
    # last index updated in inds_basisC_to_exp.
    ind_C = 0
    
    basisB_inds    = []
    row_inds       = []
    col_inds       = []
    data           = []
    for ind_B in range(len(basisB)):
        ind_exp_B = inds_basisB_to_exp[ind_B]
        for ind_A in range(len(basisA)):
            ind_exp_A = inds_basisA_to_exp[ind_A]
            key_BA    = (ind_exp_B, ind_exp_A)
            
            result_C = explored_s_constants[key_BA]
            
            if result_C is not None:
                (coeff_C, ind_exp_C) = result_C
                
                if ind_exp_C in inds_exp_to_basisC:
                    row_ind = inds_exp_to_basisC[ind_exp_C] # an already stored ind_C
                else:
                    row_ind                       = ind_C
                    inds_exp_to_basisC[ind_exp_C] = row_ind
                    inds_basisC_to_exp.append(ind_exp_C)
                    ind_C += 1
                
                col_ind = ind_A
                datum   = coeff_C # * coeff_B # Note: coeff_B is what you would multiply here if computing the Liouvillian matrix. But that is done later in build_l_matrix().
                
                # Keep track of which basisB (operator)
                # index this term comes from.
                # This is used later for constructing
                # the Liouvillian matrix.
                basisB_inds.append(ind_B)
                
                row_inds.append(row_ind)
                col_inds.append(col_ind)
                data.append(datum)
                
    basisB_inds = np.array(basisB_inds, dtype=int)
    row_inds    = np.array(row_inds, dtype=int)
    col_inds    = np.array(col_inds, dtype=int)
    data        = np.array(data, dtype=complex)
    
    s_constants_data = [basisB_inds, row_inds, col_inds, data]

    # The indices of the extended basis (basisC) used to specify
    # the extended_basis corresponding to these s_constants.
    inds_basisC_to_exp = np.array(inds_basisC_to_exp, dtype=int)

    # The index of the identity OperatorString in basisC.
    try:
        # Note: Because explored_extended_basis is shared between commutator and anticommutator
        # data, the identity could be in explored_extended_basis, but not
        # in inds_exp_to_basisC.
        indC_identity = inds_exp_to_basisC[explored_extended_basis.index(identity)]
    except KeyError:
        indC_identity = None
    # NOTE: In principle, we could have returned inds_exp_to_basisC
    # which would have contained indC_identity, but I wanted to avoid
    # storing unnecessary data. There was no place in the code that
    # ever needed inds_basisC_to_exp, but there were places that needed indC_identity.
        
    return [s_constants_data, inds_basisC_to_exp, indC_identity]
        
def build_s_constants(basisA, basisB, explored_basis, explored_extended_basis, explored_s_constants, operation_mode='commutator'):
    """Build the structure constants f_{ba}^c for the given bases
    using the data stored in explored_basis and explored_s_constants.
    """
    
    # Explore the space of OperatorStrings, starting from
    # the given Basis. Update the explored_basis
    # and explored_s_constants variables as you go.
    #
    # Notes:
    # [S_b, S_a] = \sum_c f_{ba}^c S_c
    # explored_s_constants is a dictionary mapping (b,a) to (f_{ba}^c, c)
    # where a,b,c are indices of OperatorStrings S_a, S_b, S_c in the
    # relevant explored bases.
    
    # 1. First, identify completely unexplored OperatorStrings.
    unexplored_basisA = Basis([os for os in basisA if os not in explored_basis])
    unexplored_basisB = Basis([os for os in basisB if os not in explored_basis and os not in basisA])
    
    # 2. Update the explored_basis so that these OperatorStrings can be indexed.
    explored_basis += unexplored_basisA
    explored_basis += unexplored_basisB
    
    # 3. Prepare index convertors for future use.
    [inds_basisA_to_exp, inds_exp_to_basisA] = _basis_index_convertors(basisA, explored_basis)
    [inds_basisB_to_exp, inds_exp_to_basisB] = _basis_index_convertors(basisB, explored_basis)
    
    # 4. Update the explored_s_constants data.
    _update_explored_s_constants(inds_basisA_to_exp, inds_basisB_to_exp, explored_basis, explored_extended_basis, explored_s_constants, operation_mode)
    
    # 5. Build the structure constants data with the updated data.
    [s_constants_data, inds_extended_basis, ind_identity_extended_basis] = _build_s_constants_data(basisA, basisB, inds_basisA_to_exp, inds_basisB_to_exp, explored_extended_basis, explored_s_constants)
    
    return [s_constants_data, inds_extended_basis, ind_identity_extended_basis]

# TODO: document
def build_l_matrix(s_constants_data, op, basis, inds_extended_basis):
    """Build the Liouvillian matrix (L_H)_{ca} = \sum_b J_b f_{ba}^c (or anti-Liouvillian matrix
    \\bar{L}_H) from the structure constants f_{ba}^c and the vector J_b.
    
    """
    
    if isinstance(op, np.ndarray):
        vec = op
    elif isinstance(op, Operator):
        vec = op.coeffs
    else:
        raise TypeError('Invalid op of type: {}'.format(type(op)))
    
    [basisB_inds, row_inds, col_inds, data] = s_constants_data
    
    # Update the data to include the contribution from the vector J_b.
    new_data = data * vec[basisB_inds]
    
    l_matrix = ss.csr_matrix((new_data, (row_inds, col_inds)), shape=(len(inds_extended_basis), len(basis)), dtype=np.complex)
    
    return l_matrix

# TODO: document
def print_operator(op, num_terms=20):
    """Print the largest terms of an Operator.
    """
    
    ind = 0
    for ind_s in np.argsort(np.abs(op.coeffs))[::-1]:
        coeff = op.coeffs[ind_s]
        os    = op._basis[ind_s]
        if ind < num_terms:
            print('{} {}'.format(coeff, os), flush=True)
        ind += 1

# TODO: document
def project_vector(vec1, basis_inds1, basis_inds2):
    """Project a vector into the basis of another vector.
    """

    # Projects vector1 with basis_inds1 to the basis of vector2 with basis_inds2.
    
    basis_inds2_dict = dict()
    for ind2 in range(len(basis_inds2)):
        basis_inds2_dict[basis_inds2[ind2]] = ind2

    inds_v1_proj = []
    inds_v2_proj = []
    for ind1 in range(len(vec1)):
        ind_basis = basis_inds1[ind1]
        if ind_basis in basis_inds2_dict:
            ind2 = basis_inds2_dict[ind_basis]
            inds_v1_proj.append(ind1)
            inds_v2_proj.append(ind2)

    inds_v1_proj = np.array(inds_v1_proj, dtype=int)
    inds_v2_proj = np.array(inds_v2_proj, dtype=int)

    proj_v1 = np.zeros(len(basis_inds2))
    proj_v1[inds_v2_proj] = vec1[inds_v1_proj]

    return proj_v1

# TODO: document
def project(basis, op):
    """Project the operator into the given basis.

    Note
    ----
    Creates a copy of the basis so that the projected
    operator is an independent object.
    """
    
    coeffs     = []
    inds_basis = []
    for (coeff, os) in op:
        if os in basis:
            coeffs.append(coeff)
            inds_basis.append(basis.index(os))

    inds_basis = np.array(inds_basis, dtype=int)
    coeffs     = np.array(coeffs, dtype=complex)
            
    vec             = np.zeros(len(basis), dtype=complex)
    vec[inds_basis] = coeffs

    projected_op = Operator(vec, copy.deepcopy(basis), op_type = op.op_type)
    return projected_op

def find_local_H(basis, H):
    """Find the "local" part of Hamiltonian H
    that has non-trivially commutators with the OperatorStrings
    in the given basis (because they have overlapping sites).
    Returns the local Hamiltonian Operator.

    NOTE: This function only works for Pauli strings!
    (It assumes that OperatorStrings with no overlapping
    sites do not commute, which is true only for Pauli strings,
    but not Majorana strings or Fermion strings.)
    """

    if H.op_type != 'Pauli':
        raise ValueError('This function does not work for OperatorStrings of type: {}'.format(H.op_type))

    # All of the sites that the OperatorStrings in basis have support on.
    sites_in_basis = set([orb_label for os in basis for orb_label in os.orbital_labels])

    # Note: an empty set evaluates to False. So if the intersection set in the list comprehension
    # is not empty, then you add that OperatorString to the basis.
    inds_local = np.array([ind_os for ind_os in range(len(H._basis)) if sites_in_basis.intersection(H._basis[ind_os].orbital_labels)], dtype=int)

    # The local part of the Hamiltonian H that has overlap with the sites in basis.
    H_local = Operator(H.coeffs[inds_local], Basis([H._basis[ind_os] for ind_os in inds_local]))
    
    return H_local

# TODO: document
def expand_com(H, com_residual, inds_com_extended_basis, basis, dbasis, explored_basis, explored_extended_basis, explored_com_data, truncation_size=None, verbose=False):
    """Expand the basis by commuting with the Hamiltonian H.
    Compute [H, [H, \tau]] and add the OperatorStrings with the
    largest coefficients to the basis.
    """
    
    if verbose:
        print('|Basis of \\tau|               = {}'.format(len(basis)), flush=True)
        print('|Basis of [H,\\tau]|           = {}'.format(len(com_residual)), flush=True)

    if truncation_size is not None:
        vec_size = np.minimum(len(com_residual), truncation_size)
        if verbose:
            print('|Basis of truncated [H,\\tau]| = {}'.format(vec_size), flush=True)
        
        inds_sort = np.argsort(np.abs(com_residual))[::-1]
        inds_sort = inds_sort[0:vec_size]
        t_com_residual       = com_residual[inds_sort]
        t_com_extended_basis = Basis([explored_extended_basis[inds_com_extended_basis[ind]] for ind in inds_sort])
    else:
        t_com_residual       = com_residual
        t_com_extended_basis = Basis([explored_extended_basis[ind_exp] for ind_exp in inds_com_extended_basis])

    if H.op_type == 'Pauli':
        # Consider only the terms in H that have spatial overlap
        # with the current basis, so that you do not waste
        # time and memory considering commutators between far away Pauli strings.
        H_local = find_local_H(t_com_extended_basis, H)
    else:
        H_local = H
    [s_constants_ext, inds_ext_ext_basis, _] = build_s_constants(t_com_extended_basis, H_local._basis, explored_basis, explored_extended_basis, explored_com_data, operation_mode='commutator')
    L_H_ext                                  = build_l_matrix(s_constants_ext, H_local, t_com_extended_basis, inds_ext_ext_basis)

    if verbose:
        print('|Basis of [H, [H, \\tau]]|     = {}'.format(len(inds_ext_ext_basis)), flush=True)
    
    com_H_residual = L_H_ext.dot(t_com_residual)
    
    inds_sort = np.argsort(np.abs(com_H_residual))[::-1]
    ind_add   = 0
    num_added = 0
    while num_added < dbasis and ind_add < len(inds_sort):
        os = explored_extended_basis[inds_ext_ext_basis[inds_sort[ind_add]]]
        if os not in basis:
            basis     += os
            num_added += 1
        ind_add += 1

# TODO: document
def expand_anticom(anticom_residual, inds_anticom_extended_basis, basis, dbasis, explored_extended_basis):
    """Expand the basis by anticommuting with the operator O.
    Compute {O, O}/2 = O^2 and add the OperatorStrings with the
    largest coefficients to the basis.
    """
    
    identity_os  = opstring('I', op_type=basis[0].op_type)
    ind_identity = explored_extended_basis.index(identity_os)
    
    inds_sort = np.argsort(np.abs(anticom_residual))[::-1]
    ind_add   = 0
    num_added = 0
    while num_added < dbasis and ind_add < len(inds_sort):
        ind_exp_os = inds_anticom_extended_basis[inds_sort[ind_add]]
        os         = explored_extended_basis[ind_exp_os]
        if os not in basis and ind_exp_os != ind_identity:
            basis     += os
            num_added += 1
        ind_add += 1

### Extra miscellaneous tool functions.

# TODO: document
# From https://goshippo.com/blog/measure-real-size-any-python-object/
def get_size(obj, seen=None):
    """Recursively finds size of objects. """
    
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

# Compute a finite-difference approximation
# to a gradient, given the function.
def finite_diff_gradient(f, x, eps=1e-6):
    n = len(x)
    
    grad_f = np.zeros(n)
    for i in range(n):
        dx        = np.zeros(n)
        dx[i]     = eps
        grad_f[i] = (f(x + dx) - f(x - dx))/(2.0*eps)

    return grad_f

# Compute a finite-difference approximation
# to a hessian of a function, given the gradient.
def finite_diff_hessian(grad_f, x, eps=1e-6):
    n = len(x)
    
    hess_f  = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dy          = np.zeros(n)
            dy[j]       = eps
            
            hess_f[i,j] = (grad_f(x + dy)[i] - grad_f(x - dy)[i])/(2.0*eps)

    return hess_f


