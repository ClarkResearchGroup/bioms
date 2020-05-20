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

def _explore(basis, op, explored_basis, explored_s_constants, operation_mode='commutator'):
    # Explore the space of OperatorStrings, starting from
    # the given Basis. Update the explored_basis
    # and explored_s_constants variables as you go.
    
    basisA = basis
    basisB = op._basis
    
    explored_basis += basisA

    extended_basis = Basis() # basisC

    row_inds = []
    col_inds = []
    data     = []
    for (coeff_B, os_B) in op:
        for os_A in basisA:
            key_BA = (os_B, os_A)
            try:
                (coeff_C, os_C) = explored_s_constants[key_BA]
            except KeyError:
                (coeff_C, os_C) = _operation_opstring(os_B, os_A, operation_mode=operation_mode)
                explored_s_constants[key_BA] = (coeff_C, os_C)
                
            if os_C is not None:
                extended_basis += os_C
                
                row_ind = extended_basis.index(os_C)
                col_ind = basisA.index(os_A)
                datum   = coeff_B * coeff_C
                
                row_inds.append(row_ind)
                col_inds.append(col_ind)
                data.append(datum)
      
    l_matrix = ss.csr_matrix((data, (row_inds, col_inds)), shape=(len(extended_basis), len(basis)), dtype=np.complex)

    return [l_matrix, extended_basis]

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

# TODO: document
def lmatrix(basis, H, explored_basis, explored_s_constants, operation_mode='commutator'):
    """Build the Liouvillian matrix L_H (or anti-Liouvillian matrix
    \\bar{L}_H) from the explored data.
    
    ...
    """

    return _explore(basis, H, explored_basis, explored_s_constants, operation_mode=operation_mode)

# TODO: document
def expand_com(H, com_residual, com_extended_basis, basis, dbasis, explored_basis, explored_com_data, truncation_size=None, verbose=False):
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
        t_com_extended_basis = Basis([com_extended_basis[ind] for ind in inds_sort])
    else:
        t_com_residual       = com_residual
        t_com_extended_basis = com_extended_basis
    
    [L_H_ext, ext_ext_basis] = lmatrix(t_com_extended_basis, H, explored_basis, explored_com_data, operation_mode='commutator')

    if verbose:
        print('|Basis of [H, [H, \\tau]]|     = {}'.format(len(ext_ext_basis)), flush=True)
    
    com_H_residual = L_H_ext.dot(t_com_residual)
    
    inds_sort = np.argsort(np.abs(com_H_residual))[::-1]
    ind_add   = 0
    num_added = 0
    while num_added < dbasis and ind_add < len(inds_sort):
        os = ext_ext_basis[inds_sort[ind_add]]
        if os not in basis:
            basis     += os
            num_added += 1
        ind_add += 1

# TODO: document
def expand_anticom(anticom_residual, anticom_extended_basis, basis, dbasis):
    """Expand the basis by anticommuting with the operator O.
    Compute {O, O}/2 = O^2 and add the OperatorStrings with the
    largest coefficients to the basis.
    """
    
    identity_os = opstring('I', op_type=basis[0].op_type)
    
    inds_sort = np.argsort(np.abs(anticom_residual))[::-1]
    ind_add   = 0
    num_added = 0
    while num_added < dbasis and ind_add < len(inds_sort):
        os = anticom_extended_basis[inds_sort[ind_add]]
        if os not in basis and os != identity_os:
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


