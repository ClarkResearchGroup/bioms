import sys

import numpy as np
import numpy.linalg as nla
import scipy.sparse as ss
import scipy.sparse.linalg as ssla

from qosy.operatorstring import opstring
from qosy.basis          import Basis, Operator
from qosy.algebra        import structure_constants

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

def _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data, operation_mode='commutator', allowed_labels=None):
    # Explore the space of OperatorStrings, starting from
    # the given Basis. Update the explored_basis, explored_extended_basis,
    # and explored_s_constants variables as you go.

    # First, find the part of basis that is unexplored.
    unexplored_basis = Basis([os for os in basis if os not in explored_basis])
    
    # Then explore that part.
    [unexplored_s_constants_data, unexplored_extended_basis] = structure_constants(unexplored_basis, H._basis, return_extended_basis=True, return_data_tuple=True, operation_mode=operation_mode, tol=0.0)
    
    # Update the explored Bases.
    explored_basis          += unexplored_basis
    explored_extended_basis += unexplored_extended_basis

    # Update the explored structure constants data.
    for (coeff, os) in H:
        # The new information found from exploring.
        row_inds_unexplored = [explored_extended_basis.index(unexplored_extended_basis.op_strings[ind_eb_os]) for ind_eb_os in unexplored_s_constants_data[os][0]]
        col_inds_unexplored = [explored_basis.index(unexplored_basis.op_strings[ind_b_os]) for ind_b_os in unexplored_s_constants_data[os][1]]
        data_unexplored     = unexplored_s_constants_data[os][2]

        # The old information from previous exploration.
        if os in explored_s_constants_data:
            old_row_inds = explored_s_constants_data[os][0]
            old_col_inds = explored_s_constants_data[os][1]
            old_data     = explored_s_constants_data[os][2]
        else:
            old_row_inds = []
            old_col_inds = []
            old_data     = []
            
        # The update
        explored_s_constants_data[os] = [old_row_inds + row_inds_unexplored, old_col_inds + col_inds_unexplored, old_data + data_unexplored]
    
    # From the collected information, find the
    # extended_basis corresponding to basis.
    inds_basis_to_x = [explored_basis.index(os) for os in basis]
    inds_x_to_basis = dict()
    for ind_b in range(len(basis)):
        inds_x_to_basis[inds_basis_to_x[ind_b]] = ind_b 
    
    extended_basis = Basis()
    inds_extended_basis_to_x = []
    for (coeff, os) in H:
        # If only considering a part of the Hamiltonian,
        # with OperatorStrings with the given allowed_labels, then
        # only construct the extended basis for that part.
        if allowed_labels is not None and len(set(os.orbital_labels).intersection(allowed_labels)) == 0:
            continue # TODO: check if I am doing this right
        
        [inds_x_eb, inds_x_b, _] = explored_s_constants_data[os]
        for (ind_x_eb, ind_x_b) in zip(inds_x_eb, inds_x_b):
            if ind_x_b in inds_x_to_basis and explored_extended_basis[ind_x_eb] not in extended_basis:
                extended_basis += explored_extended_basis[ind_x_eb]
                inds_extended_basis_to_x.append(ind_x_eb)
    inds_x_to_extended_basis = dict()
    for ind_eb in range(len(extended_basis)):
        inds_x_to_extended_basis[inds_extended_basis_to_x[ind_eb]] = ind_eb 

    # From the information collected from the
    # explored bases, construct the commutant matrix.
    row_inds = []
    col_inds = []
    data     = []
    for (coeff, os) in H:
        # If only considering a part of the Hamiltonian,
        # with OperatorStrings with the given allowed_labels, then
        # only construct the extended basis for that part.
        if allowed_labels is None or len(set(os.orbital_labels).intersection(allowed_labels)) != 0:
            [inds_explored_eb, inds_explored_b, explored_data] = explored_s_constants_data[os]
            
            for (ind_explored_eb, ind_explored_b, explored_datum) in zip(inds_explored_eb, inds_explored_b, explored_data):
                if ind_explored_b in inds_x_to_basis and ind_explored_eb in inds_x_to_extended_basis:
                    row_ind = inds_x_to_extended_basis[ind_explored_eb]
                    col_ind = inds_x_to_basis[ind_explored_b]
                    
                    row_inds.append(row_ind)
                    col_inds.append(col_ind)
                    data.append(coeff * explored_datum)
                
    l_matrix = ss.csr_matrix((data, (row_inds, col_inds)), shape=(len(extended_basis), len(basis)), dtype=np.complex128)

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
            print('{} {}'.format(coeff, os))
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
    """
    
    coeffs     = []
    inds_basis = []
    for (coeff, os) in op:
        if os in basis:
            coeffs.append(coeff)
            inds_basis.append(basis.index(os))

    vec = np.zeros(len(basis), dtype=complex)
    vec[inds_basis] = coeffs

    projected_op = Operator(vec, basis.op_strings, op_type = op.op_type)
    return projected_op

# TODO: document
def expand_com(H, com_residual, com_extended_basis, basis, dbasis, explored_com_data):
    """Expand the basis by commuting with the Hamiltonian H.
    Compute [H, [H, \tau]] and add the OperatorStrings with the
    largest coefficients to the basis.
    """
    
    [L_H_ext, ext_ext_basis] = lmatrix(com_extended_basis, H, explored_com_data, operation_mode='commutator')
    
    com_H_residual = L_H_ext.dot(com_residual)
    
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

def lmatrix(basis, H, explored_data, operation_mode='commutator'):
    """Build the Liouvillian matrix L_H (or anti-Liouvillian matrix
    \\bar{L}_H) from the explored data.
    
    ...
    """
    
    [explored_basis, explored_extended_basis, explored_s_constants_data] = explored_data
    
    [l_matrix, extended_basis] = _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data, operation_mode=operation_mode)
    
    # For book-keeping indices in different bases.
    inds_basis_to_x = np.array([explored_basis.index(b_os) for b_os in basis], dtype=int)
    inds_x_to_basis = dict()
    for ind_b in range(len(basis)):
        inds_x_to_basis[inds_basis_to_x[ind_b]] = ind_b

    inds_extended_basis_to_x = np.array([explored_extended_basis.index(eb_os) for eb_os in extended_basis], dtype=int)
    inds_x_to_extended_basis = dict()
    for ind_eb in range(len(extended_basis)):
        inds_x_to_extended_basis[inds_extended_basis_to_x[ind_eb]] = ind_eb 

    results = [l_matrix, extended_basis]
            
    return results


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



